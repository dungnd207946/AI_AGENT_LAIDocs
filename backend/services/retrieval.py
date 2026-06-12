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

import hashlib
import json
import logging
import re

import httpx

from ..core.config import RerankerConfig, Settings, get_settings
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
node_ids, ordered by relevance. Select 1-{max_nodes} nodes maximum.

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


def _get_document_state(doc_id: str) -> tuple[str | None, dict | None]:
    """Load current document content + tree JSON in one DB round-trip."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT content, tree_index FROM documents WHERE id=?",
            (doc_id,),
        ).fetchone()
    if not row:
        return None, None

    content = row["content"] if row["content"] else None
    tree = None
    if row["tree_index"]:
        try:
            tree = json.loads(row["tree_index"])
        except (json.JSONDecodeError, TypeError):
            tree = None
    return content, tree


# ---------------------------------------------------------------------------
# Node selection — tree reasoning (step 1 of the tree retriever)
# ---------------------------------------------------------------------------


def select_node_ids(
    tree_index: dict,
    question: str,
    settings: Settings,
    *,
    max_nodes: int = 5,
) -> list[str]:
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
        max_nodes=max(1, max_nodes),
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

    Dedicated *multimodal* units (figures with their captions/VLM descriptions
    and tables with their captions) are appended to whichever text corpus is
    used, so questions about a figure or table retrieve a focused unit instead
    of a whole section. Their ids are namespaced (``img0001``, ``tbl0001``) so
    they never collide with tree node ids or chunk ids.
    """
    content = get_document_content(doc_id)
    tree = get_tree_index(doc_id)
    return _build_retrieval_units(doc_id, content, tree)


def _build_retrieval_units(
    doc_id: str,
    content: str | None,
    tree: dict | None,
) -> list[dict]:
    """Build the shared retrieval corpus from current document state."""
    base: list[dict] = []
    if tree and tree.get("structure"):
        nodes = structure_to_list(tree["structure"])
        base = [
            {
                "unit_id": str(n.get("node_id")),
                "title": n.get("title", "Untitled"),
                "text": n.get("text", ""),
                "kind": "text",
                "heading_path": n.get("heading_path") or [],
                "path": n.get("path") or "",
            }
            for n in nodes
            if n.get("node_id") is not None and n.get("text")
        ]

    if not base and content:
        base = [
            {
                "unit_id": f"c{i + 1:04d}",
                "title": "",
                "text": chunk,
                "kind": "text",
                "heading_path": [],
                "path": "",
            }
            for i, chunk in enumerate(_chunk_text(content))
        ]

    if not base:
        return []

    if content:
        base.extend(get_multimodal_units(doc_id, content=content))
    return base


def _normalize_hash_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, list):
        return ["" if item is None else str(item) for item in value]
    return str(value)


def _compute_content_hash(content: str | None) -> str:
    payload = content if content is not None else ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compute_corpus_hash(
    doc_id: str,
    content: str | None,
    units: list[dict],
    *,
    version: int = 1,
) -> str:
    canonical_units = []
    for unit in sorted(units, key=lambda item: str(item.get("unit_id") or "")):
        canonical_units.append(
            {
                "unit_id": _normalize_hash_value(unit.get("unit_id")),
                "kind": _normalize_hash_value(unit.get("kind")),
                "title": _normalize_hash_value(unit.get("title")),
                "heading_path": _normalize_hash_value(unit.get("heading_path") or []),
                "path": _normalize_hash_value(unit.get("path")),
                "text": _normalize_hash_value(unit.get("text")),
            }
        )

    payload = {
        "version": version,
        "doc_id": doc_id,
        "content_hash": _compute_content_hash(content),
        "units": canonical_units,
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _compute_unit_hash(unit: dict) -> str:
    """Hash one retrieval unit so unchanged chunks can reuse embeddings."""
    payload = {
        "version": 1,
        "unit_id": _normalize_hash_value(unit.get("unit_id")),
        "kind": _normalize_hash_value(unit.get("kind")),
        "title": _normalize_hash_value(unit.get("title")),
        "heading_path": _normalize_hash_value(unit.get("heading_path") or []),
        "path": _normalize_hash_value(unit.get("path")),
        "text": _normalize_hash_value(unit.get("text")),
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _unit_embedding_text(unit: dict) -> str:
    return f"{unit.get('title') or ''}\n{unit.get('text') or ''}".strip()[:8000]


def _get_current_corpus(doc_id: str) -> tuple[str | None, list[dict], str]:
    """Load current document state, retrieval units, and canonical corpus hash."""
    content, tree = _get_document_state(doc_id)
    units = _build_retrieval_units(doc_id, content, tree)
    return content, units, _compute_corpus_hash(doc_id, content, units)


def build_context_from_units(units: list[dict]) -> str:
    """Assemble a bounded context string from retrieval units.

    Figures and tables are labelled distinctly so the model can cite them and
    knows it is reading an image description or tabular data, not prose.
    """
    ctx = ""
    for u in units:
        kind = u.get("kind", "text")
        title = u.get("title") or "Section"
        uid = u.get("unit_id", "?")
        text = u.get("text", "")
        if kind == "image":
            header = f"[Figure: {title} ({uid})]"
        elif kind == "table":
            header = f"[Table: {title} ({uid})]"
        else:
            header = f"[Section: {title} (node {uid})]"
        section = f"{header}\n{text}\n\n"
        if len(ctx) + len(section) > MAX_CONTEXT_CHARS:
            break
        ctx += section
    return ctx.strip()


def _evidence_preview(unit: dict, *, limit: int = 220) -> str:
    """One-line snippet of a unit's text for citation hover/preview (no newlines)."""
    text = " ".join(str(unit.get("text") or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _evidence_heading_path(unit: dict) -> list[str]:
    raw = unit.get("heading_path") or []
    if not isinstance(raw, list):
        raw = [str(raw)]
    return [str(item).strip() for item in raw if str(item).strip()]


def evidence_from_units(units: list[dict]) -> list[dict]:
    """Return stable evidence metadata for retrieved units.

    Includes ``heading_path`` and a one-line ``preview`` so the UI can render
    citation chips and jump-to-source without a second backend round trip.
    """
    evidence: list[dict] = []
    seen: set[str] = set()
    for unit in units:
        unit_id = str(unit.get("unit_id") or "")
        if not unit_id or unit_id in seen:
            continue
        seen.add(unit_id)
        evidence.append(
            {
                "unit_id": unit_id,
                "unit_hash": _compute_unit_hash(unit),
                "title": unit.get("title") or "",
                "kind": unit.get("kind") or "text",
                "heading_path": _evidence_heading_path(unit),
                "preview": _evidence_preview(unit),
            }
        )
    return evidence


def get_current_unit_hashes(doc_id: str) -> dict[str, str]:
    """Map current retrieval unit ids to their content hashes."""
    content, tree = _get_document_state(doc_id)
    units = _build_retrieval_units(doc_id, content, tree)
    return {
        str(unit.get("unit_id") or ""): _compute_unit_hash(unit)
        for unit in units
        if str(unit.get("unit_id") or "")
    }


# ---------------------------------------------------------------------------
# Multimodal units — figures (image captions / VLM descriptions) and tables
# ---------------------------------------------------------------------------

# ``![alt](url)`` — Docling/VaultPictureSerializer emit ``![Image N](/assets/..)``.
_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<url>[^)]*)\)")
# A generic, content-free alt text we should not treat as a real caption.
_GENERIC_ALT_RE = re.compile(r"^image\s*\d*$", re.IGNORECASE)
# Markdown table separator row, e.g. ``| --- | :---: |``.
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
# A line that looks like an explicit figure/table caption (multilingual).
_CAPTION_RE = re.compile(
    r"^\s*(?:#+\s*|\*+\s*)?(?:figure|fig\.?|table|chart|diagram|"
    r"hình|bảng|biểu đồ)\b",
    re.IGNORECASE,
)


def _clean_caption(line: str) -> str:
    """Strip markdown heading/emphasis markers from a candidate caption line."""
    return line.strip().lstrip("#>*_ ").strip().strip("*_").strip()


def _nearest_caption(lines: list[str], idx: int) -> str:
    """Find an explicit caption near ``lines[idx]`` (a figure/table anchor).

    Looks a few non-blank lines before and after for a line beginning with
    ``Figure``/``Table``/etc. Returns the cleaned caption text, or "".
    """
    for delta in (-1, -2, 1, 2, -3, 3):
        j = idx + delta
        if 0 <= j < len(lines):
            line = lines[j].strip()
            if line and _CAPTION_RE.match(line) and not line.startswith("|"):
                return _clean_caption(line)[:200]
    return ""


def _extract_image_units(content: str) -> list[dict]:
    """Parse figures into retrieval units: alt text + VLM description + caption.

    Skips images whose only signal is a generic ``Image N`` alt with no
    description or caption — they add noise to lexical/dense retrieval without
    being answerable. Each kept unit gets id ``img0001``, ``img0002``, ...
    """
    lines = content.split("\n")
    units: list[dict] = []
    n = 0
    for idx, line in enumerate(lines):
        m = _IMG_RE.search(line)
        if not m:
            continue
        n += 1
        alt = m.group("alt").strip()

        # Collect the VLM description block: ``> **Description:** ...`` quote
        # lines that follow the image (allowing one blank line in between).
        desc_parts: list[str] = []
        j = idx + 1
        seen_blank = False
        while j < len(lines):
            s = lines[j].strip()
            if not s:
                if seen_blank:
                    break
                seen_blank = True
                j += 1
                continue
            if s.startswith(">"):
                desc_parts.append(s.lstrip(">").strip())
                j += 1
                continue
            break
        description = " ".join(desc_parts)
        description = re.sub(r"^\**\s*description:\s*\**\s*", "", description,
                             flags=re.IGNORECASE).strip()

        caption = _nearest_caption(lines, idx)
        generic = bool(_GENERIC_ALT_RE.match(alt))
        if not description and not caption and generic:
            continue  # nothing searchable about this image

        title = caption or (alt if not generic else f"Figure {n}")
        text_parts = [p for p in (caption, alt if not generic else "", description) if p]
        text = "\n".join(text_parts)
        units.append({
            "unit_id": f"img{n:04d}",
            "title": title[:200],
            "text": text,
            "kind": "image",
            "heading_path": [],
            "path": "",
        })
    return units


def _extract_table_units(content: str) -> list[dict]:
    """Parse markdown tables into retrieval units, with nearby captions.

    A table is a run of ``|``-delimited rows whose second line is a separator
    (``| --- |``). The intact markdown table is kept as the unit text so the
    model can read the cells directly for table QA. Ids are ``tbl0001``, ...
    """
    lines = content.split("\n")
    units: list[dict] = []
    n = 0
    i = 0
    while i < len(lines):
        row = lines[i].strip()
        is_row = row.startswith("|")
        is_sep = (
            i + 1 < len(lines)
            and "|" in lines[i + 1]
            and _TABLE_SEP_RE.match(lines[i + 1].strip())
        )
        if is_row and is_sep:
            block = [lines[i]]
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("|"):
                block.append(lines[j])
                j += 1
            n += 1
            caption = _nearest_caption(lines, i)
            table_md = "\n".join(b for b in block).strip()
            title = caption or f"Table {n}"
            text = f"{caption}\n{table_md}" if caption else table_md
            units.append({
                "unit_id": f"tbl{n:04d}",
                "title": title[:200],
                "text": text,
                "kind": "table",
                "heading_path": [],
                "path": "",
            })
            i = j
        else:
            i += 1
    return units


def get_multimodal_units(doc_id: str, content: str | None = None) -> list[dict]:
    """Return dedicated figure + table units for a document.

    Parsed lazily from the stored markdown at query time (no ingest hook), so
    multimodal retrieval works for already-converted documents too.
    """
    content = content if content is not None else get_document_content(doc_id)
    if not content:
        return []
    return _extract_image_units(content) + _extract_table_units(content)


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
    """Build & persist missing/stale dense vectors for a doc. Idempotent.

    Embeddings are computed lazily on first retrieval (no ingest-time hook),
    so this is safe to call on every query. Cache validity is per retrieval
    unit, not per whole document: editing one chunk only re-embeds that chunk.
    Returns False when embeddings are unsupported/unconfigured or the document
    has no content.
    """
    cfg = settings.active_llm
    if not embeddings_supported(cfg):
        return False
    model = embed_model_name(cfg)
    if not model:
        return False

    _content_v1, units_v1, corpus_hash_v1 = _get_current_corpus(doc_id)
    if units is not None:
        units_v1 = units
        corpus_hash_v1 = _compute_corpus_hash(doc_id, _content_v1, units_v1)
    if not units_v1:
        return False

    current_units_by_id = {
        str(u.get("unit_id") or ""): u
        for u in units_v1
        if str(u.get("unit_id") or "")
    }
    current_unit_hashes = {
        unit_id: _compute_unit_hash(unit)
        for unit_id, unit in current_units_by_id.items()
    }
    with get_db() as conn:
        rows = conn.execute(
            """SELECT unit_id, unit_hash
               FROM document_embeddings
               WHERE doc_id=? AND model=?""",
            (doc_id, model),
        ).fetchall()

    cached_hashes = {str(row["unit_id"]): row["unit_hash"] or "" for row in rows}
    missing_unit_ids = [
        unit_id
        for unit_id, unit_hash in current_unit_hashes.items()
        if cached_hashes.get(unit_id) != unit_hash
    ]

    stale_unit_ids = [
        unit_id
        for unit_id in cached_hashes
        if unit_id not in current_unit_hashes
        or cached_hashes.get(unit_id) != current_unit_hashes.get(unit_id)
    ]

    if not missing_unit_ids:
        if stale_unit_ids:
            with get_db() as conn:
                conn.executemany(
                    "DELETE FROM document_embeddings WHERE doc_id=? AND model=? AND unit_id=?",
                    [(doc_id, model, unit_id) for unit_id in stale_unit_ids],
                )
        return True

    import numpy as np

    embedder = create_embeddings(cfg)
    units_to_embed = [current_units_by_id[unit_id] for unit_id in missing_unit_ids]
    texts = [_unit_embedding_text(unit) for unit in units_to_embed]
    vectors = embedder.embed_documents(texts)

    _content_v2, units_v2, corpus_hash_v2 = _get_current_corpus(doc_id)
    units_by_id_v2 = {
        str(u.get("unit_id") or ""): u
        for u in units_v2
        if str(u.get("unit_id") or "")
    }
    unit_hashes_v2 = {
        unit_id: _compute_unit_hash(unit)
        for unit_id, unit in units_by_id_v2.items()
    }
    if any(unit_hashes_v2.get(unit_id) != current_unit_hashes.get(unit_id) for unit_id in missing_unit_ids):
        logger.info("Skipping stale embedding write for doc %s after corpus change", doc_id)
        return False
    if len(vectors) != len(units_to_embed):
        logger.warning(
            "Embedding vector count mismatch for doc %s: %s vectors for %s units",
            doc_id,
            len(vectors),
            len(units_to_embed),
        )
        return False

    with get_db() as conn:
        conn.execute("BEGIN")
        if stale_unit_ids:
            conn.executemany(
                "DELETE FROM document_embeddings WHERE doc_id=? AND model=? AND unit_id=?",
                [(doc_id, model, unit_id) for unit_id in stale_unit_ids],
            )
        for u, vec in zip(units_to_embed, vectors):
            unit_id = str(u.get("unit_id") or "")
            arr = np.asarray(vec, dtype=np.float32)
            conn.execute(
                """INSERT OR REPLACE INTO document_embeddings
                   (doc_id, unit_id, title, chunk, model, corpus_hash, unit_hash, dim, vector)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc_id,
                    unit_id,
                    u.get("title") or "",
                    u.get("text") or "",
                    model,
                    corpus_hash_v2,
                    unit_hashes_v2[unit_id],
                    int(arr.shape[0]),
                    arr.tobytes(),
                ),
            )
    return True


def dense_search(doc_id: str, question: str, settings: Settings,
                 units: list[dict] | None = None,
                 top_k: int = _PER_RETRIEVER_TOP_K) -> list[str]:
    """Rank units by embedding cosine similarity; return ordered unit_ids."""
    cfg = settings.active_llm
    model = embed_model_name(cfg)
    if not model:
        return []

    content, current_units, _corpus_hash = _get_current_corpus(doc_id)
    if units is not None:
        current_units = units
    if not current_units:
        return []

    if not ensure_embedding_index(doc_id, settings, units=current_units):
        return []

    import numpy as np

    embedder = create_embeddings(cfg)
    q = np.asarray(embedder.embed_query(question), dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-8)
    current_unit_hashes = {
        str(unit.get("unit_id") or ""): _compute_unit_hash(unit)
        for unit in current_units
        if str(unit.get("unit_id") or "")
    }

    with get_db() as conn:
        rows = conn.execute(
            """SELECT unit_id, unit_hash, vector
               FROM document_embeddings
               WHERE doc_id=? AND model=?""",
            (doc_id, model),
        ).fetchall()
    if not rows:
        return []

    scored: list[tuple[str, float]] = []
    for unit_id, unit_hash, blob in rows:
        if current_unit_hashes.get(str(unit_id)) != (unit_hash or ""):
            continue
        v = np.frombuffer(blob, dtype=np.float32)
        if v.shape != qn.shape:
            continue  # dimension mismatch (stale model) — skip
        score = float(qn @ (v / (np.linalg.norm(v) + 1e-8)))
        scored.append((unit_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [uid for uid, _ in scored[:top_k]]


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------


def is_reranker_configured(cfg: RerankerConfig) -> bool:
    return bool(cfg.enabled and cfg.base_url and cfg.model)


def _build_reranker_document(unit: dict) -> str:
    heading_path = unit.get("heading_path") or []
    if not isinstance(heading_path, list):
        heading_path = [str(heading_path)]
    heading_text = " > ".join(str(item).strip() for item in heading_path if str(item).strip())
    path = str(unit.get("path") or "").strip()
    parts = [
        f"kind: {unit.get('kind') or 'text'}",
        f"title: {unit.get('title') or ''}",
        f"heading_path: {heading_text}",
        f"path: {path}",
        f"text:\n{unit.get('text') or ''}",
    ]
    return "\n".join(parts).strip()[:4000]


def rerank_units(question: str, candidate_units: list[dict], settings: Settings) -> list[str]:
    cfg = settings.active_reranker
    if not is_reranker_configured(cfg) or not candidate_units:
        return []

    payload = {
        "model": cfg.model,
        "query": question,
        "documents": [_build_reranker_document(unit) for unit in candidate_units],
        "top_n": min(max(1, cfg.top_n), len(candidate_units)),
    }
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    timeout = httpx.Timeout(cfg.timeout_s)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(cfg.base_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Invalid reranker response: missing results")

    ranked: list[tuple[int, float]] = []
    for item in data["results"]:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        score = item.get("relevance_score", 0)
        if not isinstance(idx, int) or idx < 0 or idx >= len(candidate_units):
            continue
        try:
            ranked.append((idx, float(score)))
        except (TypeError, ValueError):
            continue

    ranked.sort(key=lambda item: item[1], reverse=True)
    return [str(candidate_units[idx]["unit_id"]) for idx, _ in ranked]


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
# Graph retrieval (GraphRAG) — one more ranked list for the fusion
# ---------------------------------------------------------------------------


def graph_search(
    doc_id: str,
    question: str,
    settings: Settings,
    units: list[dict] | None = None,
    top_k: int = _PER_RETRIEVER_TOP_K,
) -> list[str]:
    """Rank units by an entity-relation graph walk from the question's entities.

    Surfaces passages connected through the knowledge graph even when no single
    passage matches lexically/densely (the multi-hop "X founded by Y who created
    Z" case). Backed by a persistent triple cache so it costs ~one LLM call per
    query. No-op (``[]``) when GraphRAG is disabled, no LLM is configured, or
    nothing connects — so it never harms the existing pipeline.
    """
    cfg = settings.active_graph_rag
    if not cfg.enabled or not is_llm_configured(settings.active_llm):
        return []

    from . import knowledge_graph as kg  # local import avoids circular import

    ids = kg.graph_augmented_units_cached(doc_id, question, settings, hops=cfg.hops)
    if not ids:
        return []

    if units is not None:
        valid = {str(u.get("unit_id") or "") for u in units}
        ids = [uid for uid in ids if uid in valid]
    limit = min(top_k, cfg.max_units) if cfg.max_units else top_k
    return ids[:limit]


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

    reranker_cfg = settings.active_reranker
    reranker_active = is_reranker_configured(reranker_cfg)
    retriever_top_k = (
        max(_FUSED_TOP_K, reranker_cfg.candidate_k)
        if reranker_active
        else _FUSED_TOP_K
    )
    ranked_lists: list[list[str]] = []

    tree = tree if tree is not None else get_tree_index(doc_id)
    tree_selected: list[str] | None = None
    if tree and tree.get("structure"):
        try:
            tree_selected = select_node_ids(
                tree,
                question,
                settings,
                max_nodes=retriever_top_k,
            )
            if tree_selected:
                ranked_lists.append(tree_selected)
        except Exception:
            logger.exception("Tree node selection failed for doc %s", doc_id)

    try:
        bm = bm25_search(doc_id, question, units=units, top_k=retriever_top_k)
        if bm:
            ranked_lists.append(bm)
    except Exception:
        logger.exception("BM25 retrieval failed for doc %s", doc_id)

    try:
        dense = dense_search(doc_id, question, settings, units=units, top_k=retriever_top_k)
        if dense:
            ranked_lists.append(dense)
    except Exception:
        logger.exception("Dense retrieval failed for doc %s", doc_id)

    try:
        graph_ids = graph_search(doc_id, question, settings, units=units, top_k=retriever_top_k)
        if graph_ids:
            ranked_lists.append(graph_ids)
    except Exception:
        logger.exception("Graph retrieval failed for doc %s", doc_id)

    if not ranked_lists:
        return [], tree_selected

    fused = rrf_fuse(ranked_lists, top_k=retriever_top_k)
    if not reranker_active or not fused:
        return fused[:_FUSED_TOP_K], tree_selected

    unit_map = {str(unit["unit_id"]): unit for unit in units if unit.get("unit_id") is not None}
    candidate_units = [unit_map[uid] for uid in fused if uid in unit_map][:retriever_top_k]
    if not candidate_units:
        return fused[:_FUSED_TOP_K], tree_selected

    try:
        reranked = rerank_units(question, candidate_units, settings)
        if reranked:
            final_top_k = min(_FUSED_TOP_K, max(1, reranker_cfg.top_n))
            return reranked[:final_top_k], tree_selected
    except Exception:
        logger.exception("Cross-encoder reranking failed for doc %s", doc_id)
    return fused[:_FUSED_TOP_K], tree_selected


# ---------------------------------------------------------------------------
# Full hybrid retrieval pipeline (single-shot)
# ---------------------------------------------------------------------------


def retrieve_context_with_evidence(
    doc_id: str,
    question: str,
    settings: Settings | None = None,
) -> tuple[str, list[dict]]:
    """Single-shot hybrid retrieval with evidence metadata."""
    settings = settings or get_settings()

    if not is_llm_configured(settings.active_llm):
        content = get_document_content(doc_id)
        return (content[:MAX_CONTEXT_CHARS] if content else ""), []

    units = get_retrieval_units(doc_id)
    if not units:
        return "", []
    unit_map = {u["unit_id"]: u for u in units}

    fused, tree_selected = hybrid_rank(doc_id, question, settings, units=units)

    # If tree reasoning was the ONLY available signal and it deliberately
    # returned nothing, treat the question as out-of-scope (preserves the
    # anti-hallucination behaviour from the pure tree-reasoning pipeline).
    if not fused:
        if tree_selected == []:
            return "", []
        content = get_document_content(doc_id)
        return (content[:MAX_CONTEXT_CHARS] if content else ""), []

    selected = [unit_map[uid] for uid in fused if uid in unit_map]
    return build_context_from_units(selected), evidence_from_units(selected)


def retrieve_context(doc_id: str, question: str, settings: Settings | None = None) -> str:
    """Single-shot hybrid retrieval → context string for the given question.

    Fuses tree reasoning + BM25 + dense embeddings via RRF. Degrades
    gracefully: any unavailable retriever is dropped; if nothing is available,
    falls back to truncated raw content. Used as the agentic loop's fallback.
    """
    context, _evidence = retrieve_context_with_evidence(doc_id, question, settings)
    return context


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


def agentic_retrieve_context_with_evidence(
    doc_id: str,
    question: str,
    settings: Settings | None = None,
    max_rounds: int = MAX_RETRIEVAL_ROUNDS,
) -> tuple[str, list[dict]]:
    """Iterative multi-hop retrieval with self-critique and evidence metadata.

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
        return retrieve_context_with_evidence(doc_id, question, settings)

    units = get_retrieval_units(doc_id)
    if not units:
        return "", []
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
            return "", []
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
        return (content[:MAX_CONTEXT_CHARS] if content else ""), []

    ordered = sorted(accumulated.items(), key=lambda x: x[1], reverse=True)
    selected = [unit_map[uid] for uid, _ in ordered[:MAX_ACCUMULATED_UNITS]]
    return build_context_from_units(selected), evidence_from_units(selected)


def agentic_retrieve_context(
    doc_id: str,
    question: str,
    settings: Settings | None = None,
    max_rounds: int = MAX_RETRIEVAL_ROUNDS,
) -> str:
    """Iterative multi-hop retrieval with self-critique."""
    context, _evidence = agentic_retrieve_context_with_evidence(
        doc_id,
        question,
        settings,
        max_rounds=max_rounds,
    )
    return context
