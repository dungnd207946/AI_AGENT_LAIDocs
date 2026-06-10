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
import re
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend, CompositeBackend
from deepagents.backends.store import StoreBackend
from langchain.agents.middleware import dynamic_prompt
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from ..core.config import LLMConfig, Settings, get_settings, LAIDOCS_HOME
from ..core.database import get_db
from ..core.vault import ensure_assets_dir
from ..services.document_store import persist_document_content
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
read, OR EDIT the document, and NEVER tell the user "no document exists" (or ask \
"which file?") based on a file listing. The ONLY way to access the document's content is the `retrieve_context` tool. \
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

## Editing the Document
You CAN edit the document — but ONLY when the user explicitly asks you to change it
(add, modify, or delete content). Editing uses two dedicated tools, NOT your filesystem tools.

**Which document you edit (READ THIS FIRST):** Edits ALWAYS target the CURRENT document — \
the very SAME one `retrieve_context` reads. There is exactly ONE document in this \
conversation. You do NOT need, and will NOT be given, a filename or file path: the edit \
tools already point at the current document. So:
- NEVER use your filesystem tools (`ls`, `glob`, `read_file`, `write_file`) to look for, \
choose, or create a file to edit — they only see your private scratch and `/memories/`, \
NEVER the user's document (see Core Rule 5).
- NEVER ask the user "which file?" and NEVER create a new file. When the user says \
"add/edit/delete ... in the file/document", they mean THIS document — edit it in place \
with the tools below. If you feel the urge to "find the file", call `retrieve_context` instead.

1. **Preview first**: ALWAYS call `preview_edit` first. It locates the exact text in the \
real document and returns the precise snippet that will change, plus the replacement.
2. **Confirm with the user**: Show the user the previewed change and ask for explicit \
confirmation. NEVER apply an edit without it. If the user has not clearly approved, do not \
call `apply_edit`.
3. **Apply**: Only after the user clearly agrees, call `apply_edit` with the SAME \
`old_string`/`new_string` you previewed.
- `old_string` must come from `retrieve_context` or `preview_edit` — do NOT include the \
`[Section: ...]` header line, and never invent text.
- If `preview_edit` reports "not found" or "not unique", call `retrieve_context` to get more \
surrounding context, then retry with a longer, more distinctive `old_string`.
- **Delete** = pass an empty `new_string`. **Add** = use an existing passage as an anchor \
and append the new content to it (e.g. `old_string` = the anchor, `new_string` = anchor + \
new text). Never overwrite anything beyond what the user asked for.

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


def _get_document_meta(doc_id: str) -> dict | None:
    """Load lightweight identity (title/filename/folder) for the current document."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT title, filename, folder FROM documents WHERE id=?", (doc_id,)
        ).fetchone()
    if not row:
        return None
    return {"title": row[0] or "", "filename": row[1] or "", "folder": row[2] or ""}


def _build_outline(structure: list[dict], depth: int = 0, acc: list[str] | None = None) -> list[str]:
    """Flatten the tree-index structure into an indented list of section titles."""
    if acc is None:
        acc = []
    for node in structure:
        title = node.get("title")
        if title:
            acc.append("  " * depth + "- " + title)
        children = node.get("nodes")
        if children:
            _build_outline(children, depth + 1, acc)
    return acc


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
# Custom Tools — Document Editing (old_string / new_string)
# ---------------------------------------------------------------------------


def _normalize_with_map(s: str) -> tuple[str, list[int]]:
    """Collapse each whitespace run to a single space, keeping a raw-offset map.

    Returns ``(normalized, offsets)`` where ``offsets[i]`` is the index in *s*
    of the first raw character backing normalized character ``i``. Used to map a
    match found in whitespace-normalized text back to the exact raw span.
    """
    chars: list[str] = []
    offsets: list[int] = []
    i, n = 0, len(s)
    while i < n:
        if s[i].isspace():
            start = i
            while i < n and s[i].isspace():
                i += 1
            chars.append(" ")
            offsets.append(start)
        else:
            chars.append(s[i])
            offsets.append(i)
            i += 1
    return "".join(chars), offsets


def _locate_in_content(content: str, old_string: str) -> tuple[int, int] | str:
    """Find the raw ``(start, end)`` span in *content* matching *old_string*.

    Tries an exact match first; if that fails, falls back to a whitespace-
    normalized match so the agent's ``old_string`` tolerates differences in
    spacing/newlines (and an accidental ``[Section: ...]`` header is stripped by
    normalization mismatch surfacing as "not found", prompting a retry).

    The match must be UNIQUE. Returns a span tuple, or an error string suitable
    for returning straight to the agent.
    """
    if not old_string:
        return "Error: old_string is empty."

    # 1) Exact match
    exact = content.count(old_string)
    if exact == 1:
        start = content.index(old_string)
        return (start, start + len(old_string))
    if exact > 1:
        return (
            f"Error: the text appears {exact} times — add more surrounding "
            "context so it identifies a single, unique location."
        )

    # 2) Whitespace-normalized match
    norm_old = re.sub(r"\s+", " ", old_string).strip()
    if not norm_old:
        return "Error: old_string is only whitespace."
    norm_content, offsets = _normalize_with_map(content)
    occurrences = [m.start() for m in re.finditer(re.escape(norm_old), norm_content)]
    if not occurrences:
        return (
            "Error: text not found in the document. Use retrieve_context to get the "
            "exact wording, then retry (do not include the '[Section: ...]' header)."
        )
    if len(occurrences) > 1:
        return (
            f"Error: the text matches {len(occurrences)} places after normalization — "
            "add more surrounding context so it is unique."
        )
    a = occurrences[0]
    b = a + len(norm_old)
    raw_start = offsets[a]
    raw_end = offsets[b - 1] + 1  # last normalized char is non-space (norm_old is stripped)
    return (raw_start, raw_end)


@tool
def preview_edit(old_string: str, new_string: str) -> str:
    """Preview an edit to the document BEFORE applying it (read-only).

    ALWAYS call this before `apply_edit`. It locates `old_string` in the real
    document and returns the EXACT snippet that would be replaced, plus the
    replacement — so you can show the user precisely what will change and ask for
    confirmation. This tool never modifies the document.

    Args:
        old_string: The exact text to find (from retrieve_context / the document).
            Do NOT include the "[Section: ...]" header line.
        new_string: The replacement text. Pass an empty string to DELETE the match.
    """
    ctx = _tool_context_var.get()
    doc_id = ctx.get("doc_id", "")
    if not doc_id:
        return "Error: Document context not configured."

    content = _get_document_content(doc_id)
    if not content:
        return "Error: Document content is empty or not yet processed."

    located = _locate_in_content(content, old_string)
    if isinstance(located, str):
        return located  # error message

    start, end = located
    matched = content[start:end]
    action = "DELETE" if new_string == "" else "REPLACE"
    return (
        f"Preview ({action}) — ask the user to confirm before calling apply_edit.\n\n"
        f"--- Text that will be removed (exact) ---\n{matched}\n"
        f"--- Replaced with ---\n{new_string if new_string else '(deleted)'}\n"
    )


@tool
async def apply_edit(old_string: str, new_string: str) -> str:
    """Apply a previewed, USER-CONFIRMED edit to the document.

    Only call this AFTER `preview_edit` and AFTER the user has explicitly approved
    the change. Pass the SAME `old_string`/`new_string` you previewed. Writes the
    new content to the document's file, database, and search index.

    Args:
        old_string: The exact text to replace (same as previewed). Do NOT include
            the "[Section: ...]" header line.
        new_string: The replacement text. Empty string deletes the match.
    """
    ctx = _tool_context_var.get()
    doc_id = ctx.get("doc_id", "")
    if not doc_id:
        return "Error: Document context not configured."

    content = _get_document_content(doc_id)
    if not content:
        return "Error: Document content is empty or not yet processed."

    located = _locate_in_content(content, old_string)
    if isinstance(located, str):
        return located  # error message

    start, end = located
    new_content = content[:start] + new_string + content[end:]
    try:
        await persist_document_content(doc_id, new_content)
    except Exception as exc:  # noqa: BLE001 — surface to agent, don't crash stream
        return f"Error applying edit: {exc}"

    # Mark this request as having edited the document, so the chat stream can
    # signal the frontend to reload. Mutating the shared dict (rather than a
    # separate ContextVar) keeps the flag visible across tool-execution contexts.
    ctx["edited"] = True
    return "Edit applied successfully. The document has been updated."


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
# Dynamic system prompt — inject the current document's identity every turn
# ---------------------------------------------------------------------------

DOC_OUTLINE_CHAR_CAP = 1500


def _build_document_system_prompt(base: str, doc_id: str) -> str:
    """Append a "Current Document" block (title, path, section outline) to *base*.

    Gives the agent persistent awareness of which document it is working on, so
    it never has to call a tool just to discover the file or ask the user.
    """
    if not doc_id:
        return base
    meta = _get_document_meta(doc_id)
    if not meta:
        return base

    lines = [
        "",
        "## Current Document (you are ALREADY working on this — do NOT look it up)",
        f"- Title: {meta['title']}",
        f"- File: {meta['folder']}/{meta['filename']}",
    ]
    tree = _get_tree_index(doc_id)
    if tree and tree.get("structure"):
        outline = "\n".join(_build_outline(tree["structure"]))
        if outline:
            if len(outline) > DOC_OUTLINE_CHAR_CAP:
                outline = outline[:DOC_OUTLINE_CHAR_CAP] + "\n  …(outline truncated)"
            lines.append("- Section outline:\n" + outline)
    lines.append(
        "Every retrieve_context / preview_edit / apply_edit call operates on THIS "
        "document — it is the only document in this conversation. Never ask the user "
        "which file, and never create a new file."
    )
    return base + "\n" + "\n".join(lines)


@dynamic_prompt
def _current_document_prompt(request) -> str:
    """Per-request system prompt: SOUL + the current document's identity.

    Reads the per-request doc_id from the contextvar set by ``set_tool_context``
    (the same mechanism the retrieval/edit tools use). Appends to whatever system
    prompt is already on the request so it composes with other middleware.
    """
    base = getattr(request, "system_prompt", None) or DOCUMENT_SOUL_PROMPT
    doc_id = _tool_context_var.get().get("doc_id", "")
    return _build_document_system_prompt(base, doc_id)


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
        tools=[retrieve_context, read_image, preview_edit, apply_edit],
        system_prompt=DOCUMENT_SOUL_PROMPT,
        middleware=[_current_document_prompt],
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
    concurrent request isolation. The ``edited`` flag starts False and is
    flipped by ``apply_edit`` so the chat stream can signal a frontend reload.
    """
    _tool_context_var.set({"doc_id": doc_id, "settings": settings, "edited": False})


def document_was_edited() -> bool:
    """Return True if apply_edit ran during the current request's context.

    Reads the shared per-request dict set by ``set_tool_context``; ``apply_edit``
    mutates that same dict in place, so the flag is visible here after streaming.
    """
    return bool(_tool_context_var.get().get("edited", False))


def reset_agent() -> None:
    """Reset the agent singleton (e.g., when settings change)."""
    global _agent
    _agent = None
