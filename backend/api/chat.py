"""Chat API endpoints for document Q&A via DeepAgents.

Endpoints:
  POST /api/chat/stream          - Stream answer (SSE)
  GET  /api/chat/history/{doc_id} - Load all display messages
  POST /api/chat/new-session/{doc_id} - Start a fresh session
  DELETE /api/chat/history/{doc_id}  - Clear all history
"""

from __future__ import annotations

import hashlib
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..core.config import get_settings
from ..core.database import get_db
from ..services.agent import (
    get_document_agent,
    set_tool_context,
    reset_agent,
    document_was_edited,
    get_retrieved_evidence,
)
from ..services.chat_history import (
    get_current_session_id,
    get_messages,
    get_messages_for_session,
    save_message,
    save_message_evidence,
    start_new_session,
    delete_messages,
)
from ..services.compactor import compact_if_needed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _content_to_text(content) -> str:
    """Normalize LangChain message content blocks into plain streamed text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    doc_id: str
    question: str
    session_id: int | None = None  # If None, use current session


class HistoryResponse(BaseModel):
    doc_id: str
    messages: list[dict]


class SessionResponse(BaseModel):
    doc_id: str
    session_id: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _get_document_content_hash(doc_id: str) -> str:
    """Hash current document content to isolate agent memory per document version."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT content FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()

    if not row:
        return "missing"

    content = row[0] or ""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


@router.post("/stream")
async def chat_stream(body: ChatRequest):
    """Ask a question about a document (Server-Sent Events stream).

    Each SSE event contains a text delta.  The stream ends with [DONE].
    The agent uses Tree Reasoning retrieval to ground answers in the document.
    Before streaming, checks if conversation history needs compaction.
    """
    settings = get_settings()
    from ..services.llm import is_llm_configured
    if not is_llm_configured(settings.active_llm):
        raise HTTPException(
            status_code=503,
            detail="LLM is not configured. Please set the LLM endpoint in Settings.",
        )

    # Determine session
    session_id = body.session_id or get_current_session_id(body.doc_id)
    content_hash = _get_document_content_hash(body.doc_id)

    # Auto-compact display history if over token threshold (best-effort, non-blocking)
    try:
        await compact_if_needed(body.doc_id, settings)
    except Exception:
        logger.exception("compact_if_needed failed; continuing without compact")

    # Set tool context so retrieve_context knows which doc to search
    set_tool_context(body.doc_id, settings)

    from ..core.telemetry import track_event_sync
    track_event_sync("chat_sent", {"doc_id": body.doc_id})

    async def _event_generator():
        full_response = ""
        try:
            agent = await get_document_agent()
            config = {
                "configurable": {
                    # Include content hash so document edits do not replay
                    # checkpointed memory from a previous document version.
                    "thread_id": f"doc-{body.doc_id}-s{session_id}-v{content_hash}",
                },
                "run_name": "document-chat",
                "metadata": {
                    "doc_id": body.doc_id,
                    "session_id": session_id,
                },
                "tags": ["ai-agent-chatbot"],
            }

            prior = get_messages_for_session(body.doc_id, session_id)
            stream_input = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Prior conversation may contain history labels. "
                            "Do not use STALE or UNVERIFIED history as evidence for document facts. "
                            "Use prior VALID history only for continuity; the latest retrieve_context output is authoritative."
                        ),
                    },
                    *[{"role": m["role"], "content": m["content"]} for m in prior],
                    {"role": "user", "content": body.question},
                ],
            }

            # astream_events v2 yields dicts — filter to AI text tokens only,
            # skipping tool-call chunks and tool-node output.
            async for chunk in agent.astream_events(
                stream_input,
                version="v2",
                config=config,
            ):
                if chunk.get("event") != "on_chat_model_stream":
                    continue

                # Skip tokens produced inside the tools node
                node = chunk.get("metadata", {}).get("langgraph_node", "")
                if node == "tools":
                    continue

                message_obj = chunk.get("data", {}).get("chunk")
                if not message_obj:
                    continue

                content = getattr(message_obj, "content", "")
                if not content or getattr(message_obj, "tool_call_chunks", None):
                    continue

                token = _content_to_text(content)
                if not token:
                    continue

                full_response += token
                escaped = token.replace("\n", "\\n")
                yield f"data: {escaped}\n\n"

        except Exception as exc:
            logger.exception("Chat stream error")
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            # Save messages to display history
            if full_response:
                try:
                    save_message(body.doc_id, session_id, "user", body.question)
                    assistant_message_id = save_message(body.doc_id, session_id, "assistant", full_response)
                    save_message_evidence(
                        assistant_message_id,
                        body.doc_id,
                        get_retrieved_evidence(),
                    )
                except Exception:
                    logger.exception("Failed to save chat history")
            # Signal the frontend to reload the document if the agent edited it
            if document_was_edited():
                yield "data: [EDITED]\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{doc_id}")
async def get_chat_history(doc_id: str) -> HistoryResponse:
    """Load all messages for a document across all sessions."""
    messages = get_messages(doc_id)
    return HistoryResponse(doc_id=doc_id, messages=messages)


@router.post("/new-session/{doc_id}")
async def new_chat_session(doc_id: str) -> SessionResponse:
    """Start a new conversation session for a document.

    The agent context is reset but all previous messages remain visible.
    """
    new_id = start_new_session(doc_id)
    return SessionResponse(doc_id=doc_id, session_id=new_id)


@router.delete("/history/{doc_id}")
async def clear_chat_history(doc_id: str) -> dict:
    """Clear all chat history for a document.

    Also resets the agent so its in-memory conversation state is discarded.
    """
    delete_messages(doc_id)
    reset_agent()
    return {"status": "ok", "doc_id": doc_id}
