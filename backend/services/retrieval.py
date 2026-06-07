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
from .llm import create_chat_model, is_llm_configured
from .tree_index import find_nodes_by_ids, remove_fields

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 12_000

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
# Context assembly
# ---------------------------------------------------------------------------


def build_context_from_nodes(nodes: list[dict]) -> str:
    """Assemble a bounded context string from selected tree nodes."""
    ctx = ""
    for node in nodes:
        title = node.get("title", "Untitled")
        node_id = node.get("node_id", "?")
        text = node.get("text", "")
        section = f"[Section: {title} (node {node_id})]\n{text}\n\n"
        if len(ctx) + len(section) > MAX_CONTEXT_CHARS:
            break
        ctx += section
    return ctx.strip()


# ---------------------------------------------------------------------------
# Node selection (step 1)
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
# Full retrieval pipeline
# ---------------------------------------------------------------------------


def retrieve_context(doc_id: str, question: str, settings: Settings | None = None) -> str:
    """Two-step retrieval → context string for the given question.

    - Tree present: select nodes, fetch their text. An explicit empty
      selection returns "" (question is out of the document's scope) so the
      generator can honestly say it found nothing rather than hallucinate
      from an irrelevant raw-text fallback.
    - No tree (e.g. document has no headings): fall back to truncated raw
      content.
    """
    settings = settings or get_settings()

    if not is_llm_configured(settings.active_llm):
        # Without an LLM we cannot do node selection; best effort is raw text.
        content = get_document_content(doc_id)
        return content[:MAX_CONTEXT_CHARS] if content else ""

    tree_index = get_tree_index(doc_id)

    if tree_index and tree_index.get("structure"):
        try:
            node_ids = select_node_ids(tree_index, question, settings)
        except Exception:
            logger.exception("Node selection failed for doc %s", doc_id)
            node_ids = None

        if isinstance(node_ids, list) and len(node_ids) == 0:
            return ""

        if node_ids:
            nodes = find_nodes_by_ids(tree_index["structure"], node_ids)
            if nodes:
                return build_context_from_nodes(nodes)

    # Fallback: no tree index at all → raw content.
    content = get_document_content(doc_id)
    return content[:MAX_CONTEXT_CHARS] if content else ""
