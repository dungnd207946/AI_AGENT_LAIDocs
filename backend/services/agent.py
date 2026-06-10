"""Document assistant agent — LangGraph ReAct agent.

Uses langgraph.prebuilt.create_react_agent which works reliably across all
LLM providers (Gemini, OpenAI-compatible, Anthropic).

Memory:
  - Conversation working memory: LangGraph MemorySaver (per-session)
  - Durable preference memory: ~/.laidocs/memories/preferences.md (read at
    agent build time; injected into the system prompt)
"""

from __future__ import annotations

import contextvars
import logging
import sqlite3
from typing import Any

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from ..core.config import Settings, get_settings, LAIDOCS_HOME
from .llm import create_chat_model
from . import retrieval

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MEMORY_DIR = LAIDOCS_HOME / "memories"
PREFERENCES_FILE = MEMORY_DIR / "preferences.md"

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
4. **Retrieval first**: ALWAYS call the retrieve_context tool before answering any question \
about the document. Never answer from memory alone.

## Response Style
- Be concise and well-structured (use headers, bullets, bold for key terms)
- Match the user's language (if they ask in Vietnamese, answer in Vietnamese)
- When the document is ambiguous, present multiple interpretations clearly
"""

# ---------------------------------------------------------------------------
# Custom Tool — Retrieval
# ---------------------------------------------------------------------------

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

    context = retrieval.agentic_retrieve_context(doc_id, question, settings)
    if context:
        return context
    return "No relevant sections found in the document for this question."


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

_checkpointer: MemorySaver | None = None
_agent: CompiledStateGraph | None = None


async def _get_checkpointer() -> MemorySaver:
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


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
        tools=[retrieve_context],
        prompt=_build_system_prompt(),
        checkpointer=checkpointer,
    )
    return _agent


def set_tool_context(doc_id: str, settings: Settings) -> None:
    """Set per-request tool context (doc_id + settings) via ContextVar."""
    _tool_context_var.set({"doc_id": doc_id, "settings": settings})


def reset_agent() -> None:
    """Reset the agent singleton (call after settings change)."""
    global _agent
    _agent = None
