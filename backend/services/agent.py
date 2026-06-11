"""Document assistant agent — LangGraph ReAct agent.

Uses langgraph.prebuilt.create_react_agent which works reliably across all
LLM providers (Gemini, OpenAI-compatible, Anthropic).

Memory:
  - Conversation memory: durable AsyncSqliteSaver checkpointer
    (~/.laidocs/data/checkpoints.db) — survives restarts, keyed per session
    via thread_id "doc-{doc_id}-s{session_id}"
  - Durable preference memory: ~/.laidocs/memories/preferences.md (read at
    agent build time; injected into the system prompt)

Tools:
  - retrieve_context — hybrid tree+BM25+dense search via the retrieval module
  - read_image       — VLM image analysis for figures/charts in documents
  - preview_edit     — dry-run document edit (show diff to user for confirmation)
  - apply_edit       — apply a user-confirmed edit to the document
"""

from __future__ import annotations

import base64
import contextvars
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiosqlite
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from ..core.config import Settings, get_settings, LAIDOCS_HOME
from ..core.database import get_db
from ..core.vault import ensure_assets_dir
from ..services.chat_history import create_markdown_export, get_messages
from ..services.document_store import persist_document_content
from .llm import create_chat_model
from . import retrieval

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
You are a Document Assistant - a faithful, precise reader of the user's documents.

## Your Identity
You exist to help users understand THEIR documents. You are not a general-purpose AI.
You are a librarian who has read every page of the document and can find any answer within it.

## Core Rules (NON-NEGOTIABLE)
1. **Document-grounded ONLY**: Every claim in your answer MUST come from the document \
context retrieved by your tools. If you cannot find the answer in the document, say so honestly.
2. **No fabrication**: NEVER invent, extrapolate, or assume information not present in the \
retrieved context. "I don't see this in the document" is always a valid answer.
3. **Cite sections**: When answering, reference the section title where you found the information. \
If the answer comes from a figure or a table, cite it explicitly (e.g. "Figure: ..." or "Table: ...") \
and, for tables, read the relevant cells directly rather than guessing.
4. **Retrieval first**: Your FIRST tool call for ANY question about the document \
MUST be `retrieve_context`. Never answer from memory alone.
5. **The document is NOT a file you can browse**: Your only way to access document content \
is `retrieve_context`. NEVER tell the user "no document exists" based on a file listing. \
If you feel the urge to "look for the file", call `retrieve_context` instead.

## Reading Images
- The document may contain images referenced in context as `![Image N](/assets/...)`.
- When the user's question concerns such an image, call `read_image` with the EXACT path \
from the context and a precise prompt describing what to read.
- Only read images that actually appear in retrieved context — never invent or guess paths.
- Treat the vision model's answer as document-grounded content; cite the image (e.g. "Image 1").

## Editing the Document
You CAN edit the document — but ONLY when the user explicitly asks you to change it.

1. **Preview first**: ALWAYS call `preview_edit` first. It shows the exact text that will \
change and asks for confirmation. NEVER apply without user approval.
2. **Confirm with the user**: Show the preview and ask for explicit confirmation.
3. **Apply**: Only after the user clearly agrees, call `apply_edit` with the SAME \
`old_string`/`new_string` you previewed.
- `old_string` must come from `retrieve_context` or `preview_edit` — never invent text.
- **Delete** = pass an empty `new_string`. **Add** = use an existing passage as anchor.

## Generating markdown exports
When the user requests to export/save/download content as Markdown, use `create_markdown_file`.

**IMPORTANT: You MUST distinguish between two VERY DIFFERENT scenarios:**

### Scenario A: User wants to export your EXISTING response
User says phrases like:
- "Export that response"
- "Save this conversation"  
- "Download your last answer"
- "Xuất câu trả lời đó"

→ Call `create_markdown_file()` with NO arguments.
→ The tool automatically uses your most recent reply.

### Scenario B: User wants you to CREATE NEW content THEN export
User says phrases like:
- "Tóm tắt và xuất file" (Summarize and export)
- "Dịch và xuất file" (Translate and export)
- "Phân tích và xuất file" (Analyze and export)
- "Tạo báo cáo và xuất file" (Create report and export)
- "Viết tóm tắt phần X và xuất file" (Write summary of section X and export)

**For Scenario B, you MUST follow this EXACT process:**

**STEP 1: READ the document**
- Call `retrieve_context(question="nội dung cần tóm tắt/dịch/phân tích")`

**STEP 2: CREATE the content**
- Based on the retrieved context, write a REAL summary/translation/analysis
- The content should be DETAILED, not just a phrase
- Example: "Phần 2.1 trình bày về... Bao gồm các nội dung chính: ..."

**STEP 3: Show the content to user**
- Output the content you created

**STEP 4: Export it**
- Call `create_markdown_file(content=<the_content_you_created>)`

**NEVER just write "Tóm tắt nội dung phần X" as the content. That is NOT a real summary!**

### Examples:

**CORRECT - Real summary:**
User: "Tóm tắt nội dung phần 2.1 và xuất file"
Assistant: [Calls retrieve_context(question="nội dung phần 2.1")]
Assistant: "Dựa vào nội dung phần 2.1, đây là bản tóm tắt:
- Điểm thứ nhất: ...
- Điểm thứ hai: ...
- Kết luận: ..."
Assistant: [Calls create_markdown_file(content="Dựa vào nội dung phần 2.1, đây là bản tóm tắt:\n- Điểm thứ nhất: ...\n- Điểm thứ hai: ...\n- Kết luận: ...")]
Assistant: "✅ Xuất file thành công! Tải tại: http://..."

**WRONG - Just a placeholder:**
User: "Tóm tắt nội dung phần 2.1 và xuất file"
Assistant: [Calls create_markdown_file()] ← WRONG! No content created.
Assistant: "Tóm tắt nội dung phần 2.1" ← WRONG! That's not a real summary!

**The content you export MUST be the ACTUAL summary/translation/analysis, not just a description of what you're going to do.**

## Response Style
- Be concise and well-structured (use headers, bullets, bold for key terms)
- Match the user's language (if they ask in Vietnamese, answer in Vietnamese)
- Match the user's language. If the user asks in Vietnamese, answer in Vietnamese and generate any exported markdown content in Vietnamese.
- Do not switch to English for a Vietnamese request unless the user explicitly asks for English.
- When the document is ambiguous, present multiple interpretations clearly
"""

# ---------------------------------------------------------------------------
# Per-request context (async-safe via ContextVar)
# ---------------------------------------------------------------------------

_tool_context_var: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "tool_ctx", default={}
)

# ---------------------------------------------------------------------------
# Helper — document content access (for editing tools)
# ---------------------------------------------------------------------------


def _get_document_content(doc_id: str) -> str | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT content FROM documents WHERE id=?", (doc_id,)
        ).fetchone()
    return row[0] if row and row[0] else None


def _normalize_with_map(s: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to single space, keeping a raw-offset map."""
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
    """Find unique span of old_string in content (exact, then whitespace-normalized).

    Returns (start, end) or an error string for the agent.
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
    raw_end = offsets[b - 1] + 1
    return (raw_start, raw_end)


# ---------------------------------------------------------------------------
# Helper — asset path resolution (for read_image)
# ---------------------------------------------------------------------------


def _resolve_asset_path(image_path: str) -> Path | None:
    """Map an image ref like /assets/<file>.png to its disk file."""
    filename = Path(image_path.split("?", 1)[0]).name
    if not filename:
        return None
    candidate = ensure_assets_dir() / filename
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Tool — Retrieval
# ---------------------------------------------------------------------------


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

    context = retrieval.agentic_retrieve_context(doc_id, question, settings)
    if context:
        return context
    return "No relevant sections found in the document for this question."


# ---------------------------------------------------------------------------
# Tool — VLM Image Reading
# ---------------------------------------------------------------------------


@tool
def read_image(image_path: str, prompt: str) -> str:
    """Read an image embedded in the document and answer a question about it.

    Use this when retrieved context contains an image reference such as
    ``![Image N](/assets/...)`` and the user's question concerns that image.
    A vision model (VLM) reads the actual image and answers your prompt.

    Args:
        image_path: The image reference exactly as it appears in the document
            context, e.g. "/assets/<doc_id>_1.png".
        prompt: A precise question or instruction describing what to read from
            the image.
    """
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
        model = create_chat_model(vlm)
        resp = model.invoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ])
        ])
        content = resp.content
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return content or "The vision model returned an empty response."
    except Exception as exc:
        return f"Error reading image: {exc}"


# ---------------------------------------------------------------------------
# Tools — Document Editing
# ---------------------------------------------------------------------------


@tool
def preview_edit(old_string: str, new_string: str) -> str:
    """Preview an edit to the document BEFORE applying it (read-only).

    ALWAYS call this before `apply_edit`. It locates `old_string` in the
    document and returns the exact snippet that would be replaced, plus the
    replacement — so you can show the user precisely what will change.

    Args:
        old_string: The exact text to find (from retrieve_context). Do NOT
            include the "[Section: ...]" header line.
        new_string: The replacement text. Pass an empty string to DELETE.
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
        return located

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

    Only call this AFTER `preview_edit` and AFTER the user has explicitly
    approved the change. Writes to the document file, database, and index.

    Args:
        old_string: The exact text to replace (same as previewed).
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
        return located

    start, end = located
    new_content = content[:start] + new_string + content[end:]
    try:
        await persist_document_content(doc_id, new_content)
    except Exception as exc:
        return f"Error applying edit: {exc}"

    # Flag this request as having edited the document so the stream sends [EDITED]
    ctx["edited"] = True
    return "Edit applied successfully. The document has been updated."


@tool
def create_markdown_file(filename: str | None = None, content: str | None = None) -> str:
    """Create a downloadable Markdown file from provided content OR latest assistant reply.

    Use this tool when the user requests ANY export operation (markdown, report, summary, etc.).
    
    BEHAVIOR:
    - If `content` is provided: Saves EXACTLY that text
    - If `content` is None: Automatically uses the MOST RECENT assistant reply from chat history
    
    This means you can call this tool WITHOUT generating new content when:
    - User says "export that last response as markdown"
    - User says "save this conversation" (will save your last reply)
    - User just wants to download what you already said
    
    WORKFLOW - WITH new content (summary/translation):
        1. Generate the content
        2. Call create_markdown_file(content=<generated_text>)
        3. Return download link
    
    WORKFLOW - WITHOUT new content (just export):
        1. Call create_markdown_file()  # content=None automatically
        2. Return download link
    
    Args:
        filename: Optional custom name (e.g., "my-report", "summary"). 
        content: Text to save. If None, uses latest assistant reply from chat history.
    
    Returns:
        Success message with download URL to the saved file.
        Using format returned by the tool allows the agent to include the answer 
    """
    ctx = _tool_context_var.get()
    doc_id = ctx.get("doc_id", "")
    if not doc_id:
        return "Error: Document context not configured."

    try:
        export_path = create_markdown_export(doc_id, filename=filename, content=content)
        
        # Determine what content was saved
        if content is None:
            # Content was auto-fetched from chat history
            messages = get_messages(doc_id)
            assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
            if not assistant_messages:
                return "Error: No assistant replies found in chat history to export."
            saved_content = assistant_messages[-1].get("content", "")
            source_note = " (exported from latest assistant reply)"
        else:
            saved_content = content
            source_note = ""
        
        # Check for empty content
        if not saved_content or not saved_content.strip():
            return "Error: Cannot export empty content. No text available to save."
        
        settings = get_settings()
        base_url = f"http://127.0.0.1:{settings.port}"
        full_url = f"{base_url}/download/{quote(export_path.name)}"
        
        # Thêm preview nội dung (300 ký tự đầu)
        content_preview = saved_content.strip()
        if len(content_preview) > 300:
            content_preview = content_preview[:300] + "..."
        
        return (
            f"✅ Xuất file thành công{source_note}!\n\n"
            f"**Download link:** {full_url}\n\n"
            f"**File name:** `{export_path.name}`\n\n"
            f"**Nội dung xem trước:**\n{content_preview}\n\n"
        )
    except ValueError as ve:
        return f"Error: {ve}"
    except Exception as exc:
        return f"Error creating markdown file: {exc}"

# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def _ensure_memory_dir() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not PREFERENCES_FILE.exists():
        PREFERENCES_FILE.write_text(
            "## User Preferences\n\n(No preferences learned yet)\n",
            encoding="utf-8",
        )


def _build_system_prompt() -> str:
    prompt = DOCUMENT_SOUL_PROMPT
    if PREFERENCES_FILE.exists():
        prefs = PREFERENCES_FILE.read_text(encoding="utf-8").strip()
        if prefs and "(No preferences learned yet)" not in prefs:
            prompt += f"\n\n## Remembered User Preferences\n{prefs}"
    return prompt


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

_checkpointer: AsyncSqliteSaver | None = None
_checkpointer_conn: aiosqlite.Connection | None = None
_agent: CompiledStateGraph | None = None


async def _get_checkpointer() -> AsyncSqliteSaver:
    """Get or create the durable SQLite conversation checkpointer.

    Backed by ~/.laidocs/data/checkpoints.db so per-session conversation
    memory survives backend restarts. Uses one long-lived aiosqlite
    connection, closed on app shutdown via close_checkpointer().
    """
    global _checkpointer, _checkpointer_conn
    if _checkpointer is None:
        CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        _checkpointer_conn = await aiosqlite.connect(str(CHECKPOINT_DB))
        _checkpointer = AsyncSqliteSaver(_checkpointer_conn)
        await _checkpointer.setup()
    return _checkpointer


async def close_checkpointer() -> None:
    """Close the durable checkpointer connection (call on app shutdown)."""
    global _checkpointer, _checkpointer_conn
    if _checkpointer_conn is not None:
        try:
            await _checkpointer_conn.close()
        except Exception:
            logger.exception("Failed to close checkpointer connection")
    _checkpointer_conn = None
    _checkpointer = None


async def get_document_agent() -> CompiledStateGraph:
    """Get or create the singleton ReAct agent."""
    global _agent
    if _agent is not None:
        return _agent

    settings = get_settings()
    _ensure_memory_dir()
    checkpointer = await _get_checkpointer()
    model = create_chat_model(settings.active_llm)

    _agent = create_react_agent(
        model=model,
        tools=[retrieve_context, read_image, preview_edit, apply_edit, create_markdown_file],
        prompt=-_build_system_prompt(),
        checkpointer=checkpointer,
    )
    return _agent


def set_tool_context(doc_id: str, settings: Settings) -> None:
    """Set per-request tool context (doc_id + settings) via ContextVar.

    Also initialises the ``edited`` flag to False so ``document_was_edited``
    returns the correct value for each new request.
    """
    _tool_context_var.set({"doc_id": doc_id, "settings": settings, "edited": False})


def document_was_edited() -> bool:
    """Return True if apply_edit ran during the current request."""
    return bool(_tool_context_var.get().get("edited", False))


def reset_agent() -> None:
    """Reset the agent singleton (call after settings change)."""
    global _agent
    _agent = None
