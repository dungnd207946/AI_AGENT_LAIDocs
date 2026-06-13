"""Document assistant agent — LangGraph ReAct agent.

Uses langgraph.prebuilt.create_react_agent which works reliably across all
LLM providers (Gemini, OpenAI-compatible, Anthropic).

Memory:
  - Conversation memory: durable AsyncSqliteSaver checkpointer
    (~/.laidocs/data/checkpoints.db) — survives restarts, keyed per global
    session via thread_id "session-{session_id}"
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
You are a Document Assistant - a faithful, precise reader of the user's selected documents.

## Your Identity
You exist to help users understand THEIR documents. You are not a general-purpose AI.
The user has selected one or more documents as the current scope; you may read and
operate ONLY within those files.

## Core Rules (NON-NEGOTIABLE)
1. **Document-grounded ONLY**: Every claim MUST come from context retrieved by your \
tools across the selected documents. If you cannot find the answer, say so honestly.
2. **No fabrication**: NEVER invent or assume information not present in retrieved \
context. "I don't see this in the selected documents" is always a valid answer.
3. **Cite file and section**: When answering, name the FILE (as shown in the \
"[File: ...]" label) and the section title where you found the information. For a \
figure or table, cite it explicitly and read the relevant cells directly.
4. **Retrieval first**: Your FIRST tool call for ANY content question MUST be \
`retrieve_context`. Never answer from memory alone.
5. **Documents are NOT files you can browse**: Your only way to access content is \
`retrieve_context`. NEVER claim "no document exists" — call `retrieve_context` instead.

## Evidence Priority
- For any factual answer about the document, the current turn's `retrieve_context` \
output is the only authoritative source.
- Conversation/session history is only for understanding the user's intent, follow-up \
references, and wording. Do NOT treat previous assistant answers as evidence about \
the current document.
- If conversation history conflicts with the latest retrieved context, ignore the \
conversation history and trust the retrieved context.
- If `retrieve_context` does not contain the answer, say that the current document \
context does not show it. Do NOT answer document facts from prior assistant messages.
- If the document may have been edited, treat previous answers as stale until they are \
confirmed by the latest `retrieve_context` output.

## Reading Images
- Context may reference images as `![Image N](/assets/...)`.
- When the question concerns such an image, call `read_image` with the EXACT path \
from the context. Only read images that appear in retrieved context.

## Connecting Concepts (multi-hop questions)
- For relational questions — how two or more things in the document connect, chains \
like "X founded by Y who created Z", or "what links A and B" — call `reason_over_graph` \
to get the explicit relation chains from the document's knowledge graph.
- `reason_over_graph` COMPLEMENTS, it does not replace, `retrieve_context`: still call \
`retrieve_context` for the supporting passages, and ground every claim in retrieved text. \
Treat the relation chains as a map of where to look, not as standalone evidence.

## Editing the Document
You CAN edit the document — but ONLY when the user explicitly asks you to change it.

1. **Infer the target file**: From the user's request and the retrieved context, \
determine WHICH in-scope file to edit. Pass it as the `file` argument (the name shown \
in the "[File: ...]" label). If it is ambiguous which file, ASK the user first.
2. **Preview first**: ALWAYS call `preview_edit` (with `file`, `old_string`, \
`new_string`) and show the result. NEVER apply without user approval.
3. **Apply**: Only after explicit approval, call `apply_edit` with the SAME `file` and \
strings you previewed.
- `old_string` must come from `retrieve_context` or `preview_edit` — never invent text.
- **Delete** = empty `new_string`. **Add** = use an existing passage as anchor.

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
- Be concise and well-structured (headers, bullets, bold for key terms)
- Match the user's language (Vietnamese in → Vietnamese out), and generate any exported markdown content in the same language
- Do not switch to English for a Vietnamese request unless the user explicitly asks for English
- When documents disagree or are ambiguous, present the interpretations clearly
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


def _resolve_scope_doc(file: str, ctx: dict[str, Any]) -> str:
    """Resolve a file label (title shown in [File: ...]) or raw doc_id to a
    doc_id within the current scope. Returns the doc_id, or an error string
    for the agent listing the available files.
    """
    doc_ids: list[str] = ctx.get("doc_ids") or []
    titles: dict[str, str] = ctx.get("doc_titles") or {}
    available = ", ".join(titles.get(d, d) for d in doc_ids) or "(none)"

    f = (file or "").strip()
    if not f:
        if len(doc_ids) == 1:
            return doc_ids[0]
        return (
            "Error: multiple files are in scope — specify which file to edit "
            f"(by its [File: ...] name). Available: {available}"
        )
    if f in doc_ids:
        return f
    matches = [d for d in doc_ids if titles.get(d, "").strip().lower() == f.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return f"Error: '{file}' matches multiple files; use the exact file name."
    return f"Error: '{file}' is not in the selected scope. Available files: {available}"


@tool
def retrieve_context(question: str) -> str:
    """Search the selected documents for sections relevant to the question.

    ALWAYS call this tool before answering any content question.
    Returns the most relevant sections across the in-scope documents, each
    labelled with its source file. If nothing relevant is found, says so.

    Args:
        question: The specific question to search for in the selected documents.
    """
    ctx = _tool_context_var.get()
    doc_ids = ctx.get("doc_ids") or []
    settings = ctx.get("settings")

    if not doc_ids or not settings:
        return "Error: Document context not configured."

    context, evidence = retrieval.agentic_retrieve_context_multi_with_evidence(
        doc_ids, question, settings
    )
    # Record retrieved units so the stream can emit citation chips ([EVIDENCE]).
    # Accumulates across multiple retrieve_context calls within one turn,
    # de-duplicated by unit_id.
    if evidence:
        existing = ctx.setdefault("retrieved_units", [])
        seen = {item.get("unit_id") for item in existing if isinstance(item, dict)}
        for item in evidence:
            if item.get("unit_id") not in seen:
                existing.append(item)
                seen.add(item.get("unit_id"))
    if context:
        return context
    return "No relevant sections found in the selected documents for this question."


# ---------------------------------------------------------------------------
# Tool — Graph-of-thought reasoning (GraphRAG)
# ---------------------------------------------------------------------------


@tool
def reason_over_graph(question: str) -> str:
    """Trace how concepts in a relational question connect across the document.

    Use this for multi-hop / "how is X related to Y" questions whose answer is
    spread across sections. Returns explicit relation chains (a knowledge-graph
    reasoning scaffold) built from the document, e.g.
    `Acme --[founded by]--> Jane --[born in]--> Paris`.

    This COMPLEMENTS `retrieve_context` — still call `retrieve_context` for the
    supporting passages and ground every claim in the retrieved text.

    Args:
        question: The relational question to map across the document.
    """
    ctx = _tool_context_var.get()
    doc_ids = ctx.get("doc_ids") or []
    settings = ctx.get("settings")

    if not doc_ids or not settings:
        return "Error: Document context not configured."

    # Sessions are global with multi-doc scope; walk each in-scope document's
    # graph and combine the non-empty scaffolds (graph_of_thought is per-doc).
    titles = ctx.get("doc_titles") or {}
    try:
        from . import knowledge_graph as kg
        scaffolds: list[str] = []
        for doc_id in doc_ids:
            scaffold = kg.graph_of_thought_cached(doc_id, question, settings)
            if not scaffold:
                continue
            if len(doc_ids) > 1:
                scaffolds.append(f"[File: {titles.get(doc_id, doc_id)}]\n{scaffold}")
            else:
                scaffolds.append(scaffold)
    except Exception:
        logger.exception("Graph-of-thought reasoning failed")
        return "No connecting relationships found in the document for this question."

    combined = "\n\n".join(scaffolds)
    if combined:
        # Expose the chain to the stream so the UI can render the reasoning path.
        ctx["reasoning_chain"] = combined
    return combined or "No connecting relationships found in the document for this question."


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
def preview_edit(file: str, old_string: str, new_string: str) -> str:
    """Preview an edit to ONE of the in-scope documents BEFORE applying it.

    ALWAYS call this before `apply_edit`. Locates `old_string` in the named
    file and returns the exact snippet that would change, plus the replacement.

    Args:
        file: Which document to edit — the file name exactly as shown in the
            "[File: ...]" label from retrieve_context. Must be one of the
            files currently in scope.
        old_string: The exact text to find (from retrieve_context). Do NOT
            include the "[File: ... | Section: ...]" header line.
        new_string: The replacement text. Pass an empty string to DELETE.
    """
    ctx = _tool_context_var.get()
    doc_id = _resolve_scope_doc(file, ctx)
    if doc_id.startswith("Error"):
        return doc_id

    content = _get_document_content(doc_id)
    if not content:
        return "Error: Document content is empty or not yet processed."

    located = _locate_in_content(content, old_string)
    if isinstance(located, str):
        return located

    start, end = located
    matched = content[start:end]
    action = "DELETE" if new_string == "" else "REPLACE"
    title = (ctx.get("doc_titles") or {}).get(doc_id, doc_id)
    return (
        f"Preview ({action}) on file '{title}' — ask the user to confirm before "
        f"calling apply_edit.\n\n"
        f"--- Text that will be removed (exact) ---\n{matched}\n"
        f"--- Replaced with ---\n{new_string if new_string else '(deleted)'}\n"
    )


@tool
async def apply_edit(file: str, old_string: str, new_string: str) -> str:
    """Apply a previewed, USER-CONFIRMED edit to one in-scope document.

    Only call AFTER `preview_edit` and AFTER the user explicitly approved.

    Args:
        file: Same file you previewed (name as in the "[File: ...]" label).
        old_string: The exact text to replace (same as previewed).
        new_string: The replacement text. Empty string deletes the match.
    """
    ctx = _tool_context_var.get()
    doc_id = _resolve_scope_doc(file, ctx)
    if doc_id.startswith("Error"):
        return doc_id

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

    ctx["edited"] = True
    ctx["edited_doc_id"] = doc_id
    title = (ctx.get("doc_titles") or {}).get(doc_id, doc_id)
    return f"Edit applied successfully to '{title}'. The document has been updated."


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
    # Context stores doc_ids (list) under global sessions; doc_id is only used
    # decoratively for the filename/header, so fall back to "chat" when no doc
    # is in scope rather than blocking the export entirely.
    doc_ids = ctx.get("doc_ids") or []
    doc_id = doc_ids[0] if doc_ids else "chat"

    try:
        export_path = create_markdown_export(doc_id, filename=filename, content=content)
        
        # Determine what content was saved
        if content is None:
            # Content was auto-fetched from chat history
            messages = get_messages()
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


# Conversation window: the checkpointer is the agent's full memory, but we only
# feed the model the last MAX_RECENT_TURNS user turns to bound token usage.
MAX_RECENT_TURNS = 10


def _trim_to_recent_turns(state: dict) -> dict:
    """pre_model_hook: cap the model's view to the last MAX_RECENT_TURNS turns.

    Returns ``llm_input_messages`` (not ``messages``), so the persisted
    checkpoint history is left intact — only what the LLM sees this step is
    trimmed. The window always starts on a HumanMessage, which keeps any
    AIMessage(tool_calls=…)/ToolMessage sequence whole (a ToolMessage is never
    orphaned at the head of the window).
    """
    messages = state["messages"]
    human_positions = [
        i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
    ]
    if len(human_positions) <= MAX_RECENT_TURNS:
        return {"llm_input_messages": messages}
    start = human_positions[-MAX_RECENT_TURNS]
    return {"llm_input_messages": messages[start:]}


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
        tools=[retrieve_context, reason_over_graph, read_image, preview_edit, apply_edit, create_markdown_file],
        prompt=_build_system_prompt(),
        checkpointer=checkpointer,
        pre_model_hook=_trim_to_recent_turns,
    )
    return _agent


def set_tool_context(
    doc_ids: list[str],
    settings: Settings,
    doc_titles: dict[str, str] | None = None,
) -> None:
    """Set per-request tool context (selected doc scope + settings) via
    ContextVar. ``doc_titles`` maps doc_id → display title so edit tools can
    resolve a file the agent names. Initialises the ``edited`` flag to False.
    """
    _tool_context_var.set({
        "doc_ids": list(doc_ids),
        "settings": settings,
        "doc_titles": doc_titles or {},
        "edited": False,
    })


def document_was_edited() -> bool:
    """Return True if apply_edit ran during the current request."""
    return bool(_tool_context_var.get().get("edited", False))


def get_retrieved_evidence() -> list[dict]:
    """Return retrieval-unit evidence collected during the current request."""
    evidence = _tool_context_var.get().get("retrieved_units", [])
    return evidence if isinstance(evidence, list) else []


def get_reasoning_chain() -> str:
    """Return the graph-of-thought chain produced this request (or '')."""
    chain = _tool_context_var.get().get("reasoning_chain", "")
    return chain if isinstance(chain, str) else ""


def reset_agent() -> None:
    """Reset the agent singleton and active checkpoint handle."""
    global _agent, _checkpointer
    _agent = None
    _checkpointer = None
