"""DeepAgents-powered document assistant with SOUL.

Replaces the stateless RAGPipeline with an agent that:
- Answers ONLY from document context (SOUL constraint)
- Maintains conversation memory within sessions (checkpointer)
- Learns user preferences across sessions (file-based memory)
- Manages context window automatically (summarization middleware)
"""

from __future__ import annotations

import contextvars
import logging
import sqlite3
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend, CompositeBackend
from deepagents.backends.store import StoreBackend
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore

from ..core.config import Settings, get_settings, LAIDOCS_HOME
from .llm import create_chat_model
from . import retrieval

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MEMORY_DIR = LAIDOCS_HOME / "memories"
PREFERENCES_FILE = MEMORY_DIR / "preferences.md"
CHECKPOINT_DB = LAIDOCS_HOME / "data" / "checkpoints.db"
STORE_DB = LAIDOCS_HOME / "data" / "memory_store.db"

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
4. **Retrieval first**: ALWAYS call the retrieve_context tool before answering any question \
about the document. Never answer from memory alone.

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

    # Agentic multi-hop retrieval: retrieve, self-critique, and chase missing
    # evidence with follow-up sub-queries before returning the fused context.
    context = retrieval.agentic_retrieve_context(doc_id, question, settings)
    if context:
        return context
    return "No relevant sections found in the document for this question."


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
_store: BaseStore | None = None
_store_conn: sqlite3.Connection | None = None


async def _get_checkpointer() -> MemorySaver:
    """Get or create the conversation checkpointer.

    Stays in-memory by design: the durable display history lives in the
    ``chat_messages`` SQLite table, so the checkpointer only needs to hold
    the active session's working memory. (A persistent ``AsyncSqliteSaver``
    was tried previously but its context-manager API is incompatible with the
    long-lived singleton agent — see docs/plans/e2e-issues.md.)
    """
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


def _get_store() -> BaseStore:
    """Get or create the durable long-term memory store.

    Backed by SQLite so learned user preferences survive restarts. Uses a
    plain ``sqlite3.Connection`` (not ``from_conn_string``) in autocommit
    mode — ``SqliteStore`` issues explicit ``BEGIN`` statements, which clash
    with the default implicit-transaction connection.
    """
    global _store, _store_conn
    if _store is None:
        STORE_DB.parent.mkdir(parents=True, exist_ok=True)
        _store_conn = sqlite3.connect(
            str(STORE_DB), check_same_thread=False, isolation_level=None
        )
        _store = SqliteStore(_store_conn)
        _store.setup()
    return _store


def _create_model(settings: Settings):
    """Create a LangChain chat model from LAIDocs settings (any provider)."""
    return create_chat_model(settings.active_llm)


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

    model = _create_model(settings)

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
        tools=[retrieve_context],
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
