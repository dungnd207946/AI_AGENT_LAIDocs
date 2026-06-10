"""DeepAgents-powered document assistant with SOUL.

Replaces the stateless RAGPipeline with an agent that:
- Answers ONLY from document context (SOUL constraint)
- Maintains conversation memory within sessions (checkpointer)
- Learns user preferences across sessions (file-based memory)
- Manages context window automatically (summarization middleware)
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend, CompositeBackend
from deepagents.backends.store import StoreBackend
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from ..core.config import LLMConfig, Settings, get_settings, LAIDOCS_HOME
from ..core.database import get_db
from ..core.vault import ensure_assets_dir
from ..services.tree_index import find_nodes_by_ids, remove_fields

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MEMORY_DIR = LAIDOCS_HOME / "memories"
PREFERENCES_FILE = MEMORY_DIR / "preferences.md"
CHECKPOINT_DB = LAIDOCS_HOME / "data" / "checkpoints.db"

# ---------------------------------------------------------------------------
# SOUL System Prompt
# ---------------------------------------------------------------------------

DOCUMENT_SOUL_PROMPT = """\
You are a Document Assistant — a faithful, precise reader of the user's documents.

## Your Identity
You exist to help users understand THEIR documents. You are not a general-purpose AI.
You are a librarian who has read every page of the document and can find any answer within it.

## Core Rules (NON-NEGOTIABLE)
1. **Document-grounded ONLY**: Every claim in your answer MUST come from the document \
context retrieved by your tools. If you cannot find the answer in the document, say so honestly.
2. **No fabrication**: NEVER invent, extrapolate, or assume information not present in the \
retrieved context. "I don't see this in the document" is always a valid answer.
3. **Cite sections**: When answering, reference the section title where you found the information.
4. **Retrieval first**: Your FIRST tool call for ANY question about the document \
MUST be `retrieve_context`. Never answer from memory alone.
5. **The document is NOT a file you can browse**: Your filesystem tools \
(`ls`, `glob`, `grep`, `read_file`) only see your private scratch and `/memories/` \
space — they will NEVER contain the user's document. NEVER use them to find, locate, \
or read the document, and NEVER tell the user "no document exists" based on a file \
listing. The ONLY way to access the document's content is the `retrieve_context` tool. \
If you feel the urge to "look for the file", call `retrieve_context` instead.

## Reading Images
- The document may contain images, referenced in the context as \
`![Image N](/assets/...)` (charts, diagrams, figures, scanned tables).
- When the user's question concerns such an image, call the `read_image` tool \
with the EXACT path from the context (e.g. `/assets/<doc_id>_1.png`) and a precise \
prompt describing what to read.
- Only read images that actually appear in the retrieved document context — \
never invent or guess image paths.
- Treat the vision model's answer as part of the document content (it is still \
document-grounded); cite the image (e.g. "Image 1") in your answer.

## Response Style
- Be concise and well-structured (use headers, bullets, bold for key terms)
- Match the user's language (if they ask in Vietnamese, answer in Vietnamese)
- When the document is ambiguous, present multiple interpretations clearly

## Memory & Learning
- Read /memories/preferences.md at the start to recall user preferences
- When you notice a clear user preference (language, detail level, format), \
save it to /memories/preferences.md for future conversations
- Only save genuine, repeated preferences — not one-off requests
"""

# ---------------------------------------------------------------------------
# Constants for retrieval
# ---------------------------------------------------------------------------

MAX_CONTEXT_CHARS = 12_000

# ---------------------------------------------------------------------------
# Helper functions (reused from rag.py)
# ---------------------------------------------------------------------------


def _get_tree_index(doc_id: str) -> dict | None:
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


def _get_document_content(doc_id: str) -> str | None:
    """Load raw markdown content for fallback."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT content FROM documents WHERE id=?", (doc_id,)
        ).fetchone()
    return row[0] if row and row[0] else None


def _build_context_from_nodes(nodes: list[dict]) -> str:
    """Build context string from selected tree nodes."""
    ctx = ""
    for node in nodes:
        title = node.get('title', 'Untitled')
        node_id = node.get('node_id', '?')
        text = node.get('text', '')
        section = f"[Section: {title} (node {node_id})]\n{text}\n\n"
        if len(ctx) + len(section) > MAX_CONTEXT_CHARS:
            break
        ctx += section
    return ctx.strip()


def _select_nodes_sync(tree_index: dict, question: str, settings: Settings) -> list[str]:
    """Ask LLM to select relevant node_ids from tree structure (sync, for thread pool).

    Uses a separate synchronous LLM call (not the agent) for node selection.
    Called via asyncio.to_thread() to avoid blocking the event loop.
    """
    import re
    from openai import OpenAI

    structure = tree_index.get('structure', [])
    structure_no_text = remove_fields(structure, fields=['text'])

    client = OpenAI(
        base_url=settings.active_llm.base_url or None,
        api_key=settings.active_llm.api_key or "sk-placeholder",
    )

    prompt = (
        "Given this document's tree structure, identify which sections are most "
        "relevant to answer the user's question. Return ONLY a JSON array of "
        "node_ids, ordered by relevance. Select 1-5 nodes maximum.\n\n"
        f"Document Structure:\n"
        f"{json.dumps(structure_no_text, ensure_ascii=False, indent=2)}\n\n"
        f"Question: {question}\n\n"
        'Return format: ["0003", "0007"]'
    )

    resp = client.chat.completions.create(
        model=settings.active_llm.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )

    raw = resp.choices[0].message.content or "[]"
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return [str(nid) for nid in parsed if isinstance(nid, (str, int))]
        except json.JSONDecodeError:
            pass
    return []


# ---------------------------------------------------------------------------
# Custom Tool — Tree Retrieval
# ---------------------------------------------------------------------------

# Per-request context using contextvars for async-safe isolation.
# Each asyncio task gets its own copy, preventing concurrent request collisions.
_tool_context_var: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "tool_ctx", default={}
)


@tool
def retrieve_context(question: str) -> str:
    """Search the document for sections relevant to the user's question.

    ALWAYS call this tool before answering any document question.
    Returns the most relevant sections with their titles and content.
    If no relevant sections are found, returns a message saying so.

    Args:
        question: The specific question to search for in the document.
    """
    ctx = _tool_context_var.get()
    doc_id = ctx.get("doc_id", "")
    settings = ctx.get("settings")

    if not doc_id or not settings:
        return "Error: Document context not configured."

    tree_index = _get_tree_index(doc_id)

    if tree_index and tree_index.get('structure'):
        node_ids = _select_nodes_sync(tree_index, question, settings)

        if isinstance(node_ids, list) and len(node_ids) == 0:
            return "No relevant sections found in the document for this question."

        nodes = find_nodes_by_ids(tree_index['structure'], node_ids)
        if nodes:
            return _build_context_from_nodes(nodes)

    # Fallback: no tree index → use raw content
    content = _get_document_content(doc_id)
    if content:
        return content[:MAX_CONTEXT_CHARS]

    return "Document content is empty or not yet processed."


# ---------------------------------------------------------------------------
# Custom Tool — VLM Image Reading
# ---------------------------------------------------------------------------


def _resolve_asset_path(image_path: str) -> Path | None:
    """Map a Markdown image ref like ``/assets/<file>.png`` to its disk file.

    Only the basename is used (defensive against path traversal); the file
    must live inside the vault assets directory. Returns None if not found.
    """
    filename = Path(image_path.split("?", 1)[0]).name
    if not filename:
        return None
    candidate = ensure_assets_dir() / filename
    return candidate if candidate.exists() else None


@tool
def read_image(image_path: str, prompt: str) -> str:
    """Read an image embedded in the document and answer a question about it.

    Use this when retrieved context contains an image reference such as
    ``![Image N](/assets/...)`` and the user's question concerns that image
    (a chart, diagram, figure, scanned table, etc.). A vision model (VLM)
    reads the actual image and answers your prompt.

    Args:
        image_path: The image reference exactly as it appears in the document
            context, e.g. "/assets/<doc_id>_1.png".
        prompt: A precise question or instruction describing what to read from
            the image.
    """
    import base64

    ctx = _tool_context_var.get()
    settings: Settings | None = ctx.get("settings")
    if not settings:
        return "Error: Document context not configured."

    vlm = settings.active_vlm
    if not vlm.base_url or not vlm.model:
        return "Error: VLM is not configured. Set it in Settings → VLM."

    file_path = _resolve_asset_path(image_path)
    if file_path is None:
        return f"Error: Image {image_path} not found."

    try:
        data = file_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"
        model = _create_model(vlm)
        resp = model.invoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ])
        ])
        content = resp.content
        if isinstance(content, list):
            # Some providers return content blocks; join text parts.
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return content or "The vision model returned an empty response."
    except Exception as exc:  # noqa: BLE001 — surface to agent, don't crash stream
        return f"Error reading image: {exc}"


# ---------------------------------------------------------------------------
# Memory initialization
# ---------------------------------------------------------------------------


def _ensure_memory_dir() -> None:
    """Create memory directory and seed preferences file if needed."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not PREFERENCES_FILE.exists():
        PREFERENCES_FILE.write_text(
            "## User Preferences\n\n(No preferences learned yet)\n",
            encoding="utf-8",
        )




# ---------------------------------------------------------------------------
# Agent Factory
# ---------------------------------------------------------------------------

_checkpointer: MemorySaver | None = None
_store: InMemoryStore | None = None


async def _get_checkpointer() -> MemorySaver:
    """Get or create the MemorySaver checkpointer."""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


def _get_store() -> InMemoryStore:
    """Get or create the in-memory store for StoreBackend."""
    global _store
    if _store is None:
        _store = InMemoryStore()
    return _store


def _create_model(cfg: LLMConfig):
    """Create a LangChain chat model from an LLM/VLM config.

    Shared by both the chat LLM (``settings.active_llm``) and the
    vision model (``settings.active_vlm``) — they speak the same
    OpenAI-compatible protocol, only base_url/api_key/model differ.
    """
    return init_chat_model(
        model=cfg.model,
        model_provider="openai",
        base_url=cfg.base_url or None,
        api_key=cfg.api_key or "sk-placeholder",
        max_retries=3,
        timeout=120,
    )


_agent: CompiledStateGraph | None = None


async def get_document_agent() -> CompiledStateGraph:
    """Get or create the singleton DeepAgent instance.

    The agent is created once and reused. Per-request state (doc_id, settings)
    is passed via thread-local _tool_context and LangGraph config.
    """
    global _agent
    if _agent is not None:
        return _agent

    settings = get_settings()
    _ensure_memory_dir()
    checkpointer = await _get_checkpointer()
    store = _get_store()

    model = _create_model(settings.active_llm)

    # Backend: CompositeBackend routes /memories/ writes to LangGraph Store
    # for persistence, default StateBackend for ephemeral scratch files.
    # Per Context7 docs: pass store= to create_deep_agent, backends
    # can be instantiated without runtime when store is passed separately.
    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend()},
    )

    _agent = create_deep_agent(
        model=model,
        tools=[retrieve_context, read_image],
        system_prompt=DOCUMENT_SOUL_PROMPT,
        memory=["/memories/preferences.md"],
        backend=backend,
        checkpointer=checkpointer,
        store=store,
        name="document-assistant",
    )

    return _agent


def set_tool_context(doc_id: str, settings: Settings) -> None:
    """Set the tool context for the current async task.

    Must be called before invoking the agent so the retrieve_context
    tool knows which document to search. Uses contextvars for safe
    concurrent request isolation.
    """
    _tool_context_var.set({"doc_id": doc_id, "settings": settings})


def reset_agent() -> None:
    """Reset the agent singleton (e.g., when settings change)."""
    global _agent
    _agent = None
