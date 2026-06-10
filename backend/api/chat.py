"""Chat API endpoints for document Q&A via DeepAgents.

Endpoints:
  POST /api/chat/stream          - Stream answer (SSE)
  GET  /api/chat/history/{doc_id} - Load all display messages
  POST /api/chat/new-session/{doc_id} - Start a fresh session
  DELETE /api/chat/history/{doc_id}  - Clear all history
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..core.config import get_settings
from ..services.agent import get_document_agent, set_tool_context, reset_agent
from ..services.chat_history import (
    get_current_session_id,
    get_messages,
    save_message,
    start_new_session,
    delete_messages,
)

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


@router.post("/stream")
async def chat_stream(body: ChatRequest):
    """Ask a question about a document (Server-Sent Events stream).

    Each SSE event contains a text delta.  The stream ends with [DONE].
    The agent uses Tree Reasoning retrieval to ground answers in the document.
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
                    "thread_id": f"doc-{body.doc_id}-s{session_id}",
                }
            }

            # Provide preferences file content if it exists
            from ..services.agent import PREFERENCES_FILE
            files = {}
            if PREFERENCES_FILE.exists():
                files["/memories/preferences.md"] = {
                    "content": PREFERENCES_FILE.read_text(encoding="utf-8"),
                    "encoding": "utf-8",
                }

            stream_input = {
                "messages": [{"role": "user", "content": body.question}],
            }
            if files:
                stream_input["files"] = files

            # Use v2 streaming format (dict-based) per LangGraph docs
            async for chunk in agent.astream(
                stream_input,
                stream_mode="messages",
                subgraphs=True,
                version="v2",
                config=config,
            ):
                if not isinstance(chunk, dict):
                    continue

                # Only process message-type chunks
                if chunk.get("type") != "messages":
                    continue

                data = chunk.get("data")
                if not isinstance(data, (list, tuple)) or len(data) != 2:
                    continue

                message_obj, metadata = data

                # Only emit AI content tokens from the main agent
                # Skip tool calls, subagent output, and summarization
                is_subagent = any(
                    s.startswith("tools:") for s in chunk.get("ns", ())
                )
                if is_subagent:
                    continue

                if (
                    hasattr(message_obj, 'type') and message_obj.type in ("ai", "AIMessageChunk")
                    and hasattr(message_obj, 'content') and message_obj.content
                    and not getattr(message_obj, 'tool_call_chunks', None)
                ):
                    token = _content_to_text(message_obj.content)
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
                    save_message(body.doc_id, session_id, "assistant", full_response)
                except Exception:
                    logger.exception("Failed to save chat history")
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
