"""Document retrieval — PageIndex tree-reasoning RAG.

Single home for the two-step retrieval used by both the chat agent's
``retrieve_context`` tool and any direct Q&A path:

  1. The LLM reads the document's tree structure (titles + summaries, no body
     text) and selects the most relevant node ids.
  2. The body text of those nodes is fetched and assembled into a context
     string bounded by ``MAX_CONTEXT_CHARS``.

Previously this logic was duplicated verbatim between ``rag.py`` and
``agent.py``; it now lives here so there is one implementation to maintain,
and all LLM calls go through the provider-agnostic factory in ``llm.py``.
"""

from __future__ import annotations

import json
import logging
import re

from ..core.config import Settings, get_settings
from ..core.database import get_db
from .llm import (
    create_chat_model,
    create_embeddings,
    embed_model_name,
    embeddings_supported,
    is_llm_configured,
)
from .tree_index import remove_fields, structure_to_list

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 12_000

# Hybrid-retrieval knobs.
_PER_RETRIEVER_TOP_K = 10  # candidates each retriever contributes to fusion
_FUSED_TOP_K = 8           # units kept after RRF fusion
_RRF_K = 60                # RRF damping constant (standard default)
_CHUNK_TARGET_CHARS = 1000  # chunk size when a document has no heading tree

_NODE_SELECT_PROMPT = """\
Given this document's tree structure, identify which sections are most \
relevant to answer the user's question. Return ONLY a JSON array of \
node_ids, ordered by relevance. Select 1-5 nodes maximum.

Document Structure:
{structure}

Question: {question}

Return format: ["0003", "0007"]"""


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def get_tree_index(doc_id: str) -> dict | None:
    """Load the tree index JSON for a document from SQLite."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT tree_index FROM documents WHERE id=?", (doc_id,)
        ).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def get_document_content(doc_id: str) -> str | None:
    """Load raw markdown content (used as fallback when there is no tree)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT content FROM documents WHERE id=?", (doc_id,)
        ).fetchone()
    return row[0] if row and row[0] else None


# ---------------------------------------------------------------------------
# Node selection — tree reasoning (step 1 of the tree retriever)
# ---------------------------------------------------------------------------


def select_node_ids(tree_index: dict, question: str, settings: Settings) -> list[str]:
    """Ask the LLM which node_ids are relevant to the question.

    Synchronous on purpose: it is called from inside the LangChain ``@tool``
    (which already runs in a worker thread) and from thread-pool executors.
    Returns an empty list if the LLM deems nothing relevant or on parse
    failure — callers distinguish "empty selection" from "no tree".
    """
    cfg = settings.active_llm
    structure = tree_index.get("structure", [])
    # Send tree WITHOUT body text — only titles, summaries, node_ids.
    structure_no_text = remove_fields(structure, fields=["text"])

    prompt = _NODE_SELECT_PROMPT.format(
        structure=json.dumps(structure_no_text, ensure_ascii=False, indent=2),
        question=question,
    )

    model = create_chat_model(cfg, temperature=0, max_tokens=200)
    resp = model.invoke([{"role": "user", "content": prompt}])
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)

    match = re.search(r"\[.*?\]", raw or "[]", re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return [str(nid) for nid in parsed if isinstance(nid, (str, int))]
        except json.JSONDecodeError:
            pass
    return []


# ---------------------------------------------------------------------------
# Retrieval units — the shared corpus for BM25 and dense retrieval
# ---------------------------------------------------------------------------


def _chunk_text(text: str, target: int = _CHUNK_TARGET_CHARS) -> list[str]:
    """Split text into ~target-sized chunks on paragraph boundaries."""
    paras = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    cur = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if cur and len(cur) + len(p) > target:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def get_retrieval_units(doc_id: str) -> list[dict]:
    """Return the unit corpus for a document, shared by all retrievers.

    Prefers tree nodes (unit_id = node_id) so BM25/dense/tree all rank the
    same units and fuse cleanly. Falls back to paragraph chunks for documents
    with no heading tree (unit_id = "c0001", ...).
    """
    tree = get_tree_index(doc_id)
    if tree and tree.get("structure"):
        nodes = structure_to_list(tree["structure"])
        units = [
            {
                "unit_id": str(n.get("node_id")),
                "title": n.get("title", "Untitled"),
                "text": n.get("text", ""),
            }
            for n in nodes
            if n.get("node_id") is not None and n.get("text")
        ]
        if units:
            return units

    content = get_document_content(doc_id)
    if not content:
        return []
    return [
        {"unit_id": f"c{i + 1:04d}", "title": "", "text": chunk}
        for i, chunk in enumerate(_chunk_text(content))
    ]


def build_context_from_units(units: list[dict]) -> str:
    """Assemble a bounded context string from retrieval units."""
    ctx = ""
    for u in units:
        title = u.get("title") or "Section"
        uid = u.get("unit_id", "?")
        text = u.get("text", "")
        section = f"[Section: {title} (node {uid})]\n{text}\n\n"
        if len(ctx) + len(section) > MAX_CONTEXT_CHARS:
            break
        ctx += section
    return ctx.strip()


# ---------------------------------------------------------------------------
# Lexical retrieval (BM25)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def bm25_search(doc_id: str, question: str, units: list[dict] | None = None,
                top_k: int = _PER_RETRIEVER_TOP_K) -> list[str]:
    """Rank units by BM25 lexical relevance; return ordered unit_ids."""
    units = units if units is not None else get_retrieval_units(doc_id)
    if not units:
        return []
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed; skipping lexical retrieval")
        return []

    corpus = [_tokenize(f"{u['title']} {u['text']}") for u in units]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(question))
    ranked = sorted(zip(units, scores), key=lambda x: x[1], reverse=True)

    positive = [u["unit_id"] for u, s in ranked if s > 0][:top_k]
    if positive:
        return positive
    # Tiny corpora: every term gets negative IDF (it appears in most/all
    # units), so nothing scores > 0. Fall back to any unit with non-zero
    # overlap — RRF only needs the rank order, not the score's sign.
    return [u["unit_id"] for u, s in ranked if s != 0][:top_k]


# ---------------------------------------------------------------------------
# Dense retrieval (Gemini / OpenAI embeddings, lazily indexed)
# ---------------------------------------------------------------------------


def ensure_embedding_index(doc_id: str, settings: Settings,
                           units: list[dict] | None = None) -> bool:
    """Build & persist the dense index for a doc if missing. Idempotent.

    Embeddings are computed lazily on first retrieval (no ingest-time hook),
    so this is safe to call on every query. Returns False when embeddings are
    unsupported/unconfigured or the document has no content.
    """
    cfg = settings.active_llm
    if not embeddings_supported(cfg):
        return False
    model = embed_model_name(cfg)
    if not model:
        return False

    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM document_embeddings WHERE doc_id=? AND model=?",
            (doc_id, model),
        ).fetchone()
    if row and row[0]:
        return True  # already indexed with this model

    units = units if units is not None else get_retrieval_units(doc_id)
    if not units:
        return False

    import numpy as np

    embedder = create_embeddings(cfg)
    texts = [f"{u['title']}\n{u['text']}".strip()[:8000] for u in units]
    vectors = embedder.embed_documents(texts)

    with get_db() as conn:
        # Clear any stale rows (e.g. from a previous embedding model) first.
        conn.execute("DELETE FROM document_embeddings WHERE doc_id=?", (doc_id,))
        for u, vec in zip(units, vectors):
            arr = np.asarray(vec, dtype=np.float32)
            conn.execute(
                """INSERT OR REPLACE INTO document_embeddings
                   (doc_id, unit_id, title, chunk, model, dim, vector)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, u["unit_id"], u["title"], u["text"], model,
                 int(arr.shape[0]), arr.tobytes()),
            )
    return True


def dense_search(doc_id: str, question: str, settings: Settings,
                 units: list[dict] | None = None,
                 top_k: int = _PER_RETRIEVER_TOP_K) -> list[str]:
    """Rank units by embedding cosine similarity; return ordered unit_ids."""
    cfg = settings.active_llm
    if not ensure_embedding_index(doc_id, settings, units=units):
        return []

    import numpy as np

    embedder = create_embeddings(cfg)
    q = np.asarray(embedder.embed_query(question), dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-8)

    with get_db() as conn:
        rows = conn.execute(
            "SELECT unit_id, vector FROM document_embeddings WHERE doc_id=?",
            (doc_id,),
        ).fetchall()
    if not rows:
        return []

    scored: list[tuple[str, float]] = []
    for unit_id, blob in rows:
        v = np.frombuffer(blob, dtype=np.float32)
        if v.shape != qn.shape:
            continue  # dimension mismatch (stale model) — skip
        score = float(qn @ (v / (np.linalg.norm(v) + 1e-8)))
        scored.append((unit_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [uid for uid, _ in scored[:top_k]]


# ---------------------------------------------------------------------------
# Rank fusion
# ---------------------------------------------------------------------------


def rrf_fuse(ranked_lists: list[list[str]], k: int = _RRF_K,
             top_k: int = _FUSED_TOP_K) -> list[str]:
    """Reciprocal Rank Fusion of multiple ranked unit_id lists."""
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, uid in enumerate(lst):
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [uid for uid, _ in fused[:top_k]]


# ---------------------------------------------------------------------------
# Hybrid ranking (single query → fused unit_ids)
# ---------------------------------------------------------------------------


def hybrid_rank(
    doc_id: str,
    question: str,
    settings: Settings,
    units: list[dict] | None = None,
    tree: dict | None = None,
) -> tuple[list[str], list[str] | None]:
    """Rank units for one query by fusing tree + BM25 + dense signals.

    Returns ``(fused_unit_ids, tree_selected)`` where ``tree_selected`` is the
    raw tree-reasoning selection (``[]`` = explicitly nothing relevant,
    ``None`` = tree unavailable/errored) so callers can preserve out-of-scope
    semantics. Any retriever that is unavailable or errors is dropped.
    """
    units = units if units is not None else get_retrieval_units(doc_id)
    if not units:
        return [], None

    ranked_lists: list[list[str]] = []

    tree = tree if tree is not None else get_tree_index(doc_id)
    tree_selected: list[str] | None = None
    if tree and tree.get("structure"):
        try:
            tree_selected = select_node_ids(tree, question, settings)
            if tree_selected:
                ranked_lists.append(tree_selected)
        except Exception:
            logger.exception("Tree node selection failed for doc %s", doc_id)

    try:
        bm = bm25_search(doc_id, question, units=units)
        if bm:
            ranked_lists.append(bm)
    except Exception:
        logger.exception("BM25 retrieval failed for doc %s", doc_id)

    try:
        dense = dense_search(doc_id, question, settings, units=units)
        if dense:
            ranked_lists.append(dense)
    except Exception:
        logger.exception("Dense retrieval failed for doc %s", doc_id)

    if not ranked_lists:
        return [], tree_selected
    return rrf_fuse(ranked_lists), tree_selected


# ---------------------------------------------------------------------------
# Full hybrid retrieval pipeline (single-shot)
# ---------------------------------------------------------------------------


def retrieve_context(doc_id: str, question: str, settings: Settings | None = None) -> str:
    """Single-shot hybrid retrieval → context string for the given question.

    Fuses tree reasoning + BM25 + dense embeddings via RRF. Degrades
    gracefully: any unavailable retriever is dropped; if nothing is available,
    falls back to truncated raw content. Used as the agentic loop's fallback.
    """
    settings = settings or get_settings()

    if not is_llm_configured(settings.active_llm):
        content = get_document_content(doc_id)
        return content[:MAX_CONTEXT_CHARS] if content else ""

    units = get_retrieval_units(doc_id)
    if not units:
        return ""
    unit_map = {u["unit_id"]: u for u in units}

    fused, tree_selected = hybrid_rank(doc_id, question, settings, units=units)

    # If tree reasoning was the ONLY available signal and it deliberately
    # returned nothing, treat the question as out-of-scope (preserves the
    # anti-hallucination behaviour from the pure tree-reasoning pipeline).
    if not fused:
        if tree_selected == []:
            return ""
        content = get_document_content(doc_id)
        return content[:MAX_CONTEXT_CHARS] if content else ""

    selected = [unit_map[uid] for uid in fused if uid in unit_map]
    return build_context_from_units(selected)


# ---------------------------------------------------------------------------
# Agentic iterative retrieval (multi-hop, self-critique)
# ---------------------------------------------------------------------------

# Loop bounds. Gemini Flash makes a few extra calls cheap; cap rounds so a
# pathological question can't run away.
MAX_RETRIEVAL_ROUNDS = 3
MAX_FOLLOWUPS_PER_ROUND = 2
MAX_ACCUMULATED_UNITS = 12

_CRITIQUE_PROMPT = """\
You judge whether retrieved document context is SUFFICIENT to fully answer a \
question, and if not, what to search for next.

Question:
{question}

Retrieved context so far:
{context}

Respond with ONLY a JSON object, no prose:
{{"sufficient": true|false, "missing": "<what is still needed, or empty>", \
"followups": ["<specific search query>", ...]}}

Rules:
- sufficient=true if the context already contains everything needed to answer.
- If the document clearly does not cover the topic, set sufficient=true and \
followups=[] (do not invent searches).
- If false, followups must be 1-{max_followups} SPECIFIC search queries \
targeting the missing information (not rephrasings of the original question).
"""


def _critique(question: str, context: str, settings: Settings) -> dict:
    """Ask the LLM whether `context` suffices; suggest follow-up sub-queries.

    Returns a dict with keys ``sufficient`` (bool) and ``followups``
    (list[str]). On any error, defaults to sufficient=True (stop the loop)
    so retrieval never hangs on a flaky critique call.
    """
    prompt = _CRITIQUE_PROMPT.format(
        question=question,
        context=context or "(nothing retrieved yet)",
        max_followups=MAX_FOLLOWUPS_PER_ROUND,
    )
    try:
        model = create_chat_model(settings.active_llm, temperature=0, max_tokens=300)
        resp = model.invoke([{"role": "user", "content": prompt}])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not match:
            return {"sufficient": True, "followups": []}
        data = json.loads(match.group())
        followups = data.get("followups") or []
        followups = [str(f).strip() for f in followups if str(f).strip()]
        return {
            "sufficient": bool(data.get("sufficient", True)),
            "followups": followups[:MAX_FOLLOWUPS_PER_ROUND],
        }
    except Exception:
        logger.exception("Retrieval critique failed; treating as sufficient")
        return {"sufficient": True, "followups": []}


def agentic_retrieve_context(
    doc_id: str,
    question: str,
    settings: Settings | None = None,
    max_rounds: int = MAX_RETRIEVAL_ROUNDS,
) -> str:
    """Iterative multi-hop retrieval with self-critique.

    Round 1 retrieves for the original question. After each round the LLM
    critiques the accumulated evidence; if insufficient, its follow-up
    sub-queries drive the next round. Evidence accumulates across rounds and
    sub-queries, scored by best RRF rank seen, then the top units are
    assembled into the final context.

    Terminates when the critique is satisfied, no new units are found, no
    follow-ups are produced, or ``max_rounds`` is reached. Falls back to the
    single-shot path when an LLM is not configured.
    """
    settings = settings or get_settings()

    if not is_llm_configured(settings.active_llm):
        return retrieve_context(doc_id, question, settings)

    units = get_retrieval_units(doc_id)
    if not units:
        return ""
    unit_map = {u["unit_id"]: u for u in units}
    tree = get_tree_index(doc_id)

    # unit_id -> best fusion score seen across all rounds/sub-queries.
    accumulated: dict[str, float] = {}
    seen_queries: set[str] = set()
    first_round_empty_tree_only = False

    queries = [question]
    for round_idx in range(max_rounds):
        new_units_this_round = 0
        for q in queries:
            qkey = q.strip().lower()
            if not qkey or qkey in seen_queries:
                continue
            seen_queries.add(qkey)

            fused, tree_selected = hybrid_rank(
                doc_id, q, settings, units=units, tree=tree
            )
            if round_idx == 0 and q == question and not fused and tree_selected == []:
                first_round_empty_tree_only = True

            for rank, uid in enumerate(fused):
                if uid not in unit_map:
                    continue
                score = 1.0 / (_RRF_K + rank + 1)
                if uid not in accumulated:
                    new_units_this_round += 1
                if score > accumulated.get(uid, 0.0):
                    accumulated[uid] = score

        # Out-of-scope: first round produced only an empty tree selection and
        # nothing else — honour the anti-hallucination contract.
        if not accumulated and first_round_empty_tree_only:
            return ""
        if not accumulated:
            break

        # Build current best-effort context for the critic.
        ordered = sorted(accumulated.items(), key=lambda x: x[1], reverse=True)
        selected = [unit_map[uid] for uid, _ in ordered[:MAX_ACCUMULATED_UNITS]]
        context = build_context_from_units(selected)

        # Last round: no point critiquing, we won't retrieve again.
        if round_idx == max_rounds - 1:
            break

        verdict = _critique(question, context, settings)
        if verdict["sufficient"] or not verdict["followups"]:
            break
        if new_units_this_round == 0:
            break  # converged — extra rounds won't add evidence

        queries = verdict["followups"]

    if not accumulated:
        # Nothing from any retriever — fall back to raw content.
        content = get_document_content(doc_id)
        return content[:MAX_CONTEXT_CHARS] if content else ""

    ordered = sorted(accumulated.items(), key=lambda x: x[1], reverse=True)
    selected = [unit_map[uid] for uid, _ in ordered[:MAX_ACCUMULATED_UNITS]]
    return build_context_from_units(selected)
