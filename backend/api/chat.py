"""Chat API — multi-document Q&A via a LangGraph ReAct agent.

Endpoints:
  POST   /api/chat/stream              - Stream answer over selected doc scope (SSE)
  GET    /api/chat/history             - Load all display messages (all sessions)
  POST   /api/chat/new-session         - Start a fresh global session
  DELETE /api/chat/history             - Clear all history
  DELETE /api/chat/session/{session_id} - Delete one session
"""

from __future__ import annotations

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
)
from ..services.chat_history import (
    get_current_session_id,
    get_messages,
    save_message,
    start_new_session,
    delete_messages,
    delete_session,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    doc_ids: list[str]
    question: str
    session_id: int | None = None  # If None, use current global session


class HistoryResponse(BaseModel):
    messages: list[dict]


class SessionResponse(BaseModel):
    session_id: int


def _doc_titles(doc_ids: list[str]) -> dict[str, str]:
    """Map each scoped doc_id → display title (title → filename → id)."""
    if not doc_ids:
        return {}
    placeholders = ",".join("?" * len(doc_ids))
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT id, COALESCE(title, filename, id) FROM documents "
            f"WHERE id IN ({placeholders})",
            tuple(doc_ids),
        ).fetchall()
    return {r[0]: r[1] for r in rows}


@router.post("/stream")
async def chat_stream(body: ChatRequest):
    """Ask a question grounded in the selected documents (SSE stream)."""
    settings = get_settings()
    from ..services.llm import is_llm_configured
    if not is_llm_configured(settings.active_llm):
        raise HTTPException(
            status_code=503,
            detail="LLM is not configured. Please set the LLM endpoint in Settings.",
        )
    if not body.doc_ids:
        raise HTTPException(status_code=400, detail="No documents selected for chat scope.")

    session_id = body.session_id or get_current_session_id()

    titles = _doc_titles(body.doc_ids)
    set_tool_context(body.doc_ids, settings, titles)

    from ..core.telemetry import track_event_sync
    track_event_sync("chat_sent", {"doc_ids": body.doc_ids})

    async def _event_generator():
        full_response = ""
        try:
            agent = await get_document_agent()
            config = {
                "configurable": {
                    # thread_id is per global session → AsyncSqliteSaver is the
                    # agent's memory: it replays the full conversation on resume,
                    # independent of doc scope. The pre_model_hook caps what the
                    # model actually sees to the last MAX_RECENT_TURNS turns.
                    "thread_id": f"session-{session_id}",
                },
                "run_name": "document-chat",
                "metadata": {
                    "doc_ids": body.doc_ids,
                    "session_id": session_id,
                },
                "tags": ["ai-agent-chatbot"],
            }

            # Only the new question is sent; prior turns live in the checkpointer
            # (no manual prior-injection → no double-feeding the conversation).
            stream_input = {
                "messages": [{"role": "user", "content": body.question}],
            }

            async for chunk in agent.astream_events(
                stream_input, version="v2", config=config,
            ):
                if chunk.get("event") != "on_chat_model_stream":
                    continue
                node = chunk.get("metadata", {}).get("langgraph_node", "")
                if node == "tools":
                    continue
                message_obj = chunk.get("data", {}).get("chunk")
                if not message_obj:
                    continue
                content = getattr(message_obj, "content", "")
                if not content or getattr(message_obj, "tool_call_chunks", None):
                    continue
                if isinstance(content, list):
                    token = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                else:
                    token = content
                full_response += token
                escaped = token.replace("\n", "\\n")
                yield f"data: {escaped}\n\n"

        except Exception as exc:
            logger.exception("Chat stream error")
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            if full_response:
                try:
                    save_message(session_id, "user", body.question)
                    save_message(session_id, "assistant", full_response)
                except Exception:
                    logger.exception("Failed to save chat history")
            if document_was_edited():
                yield "data: [EDITED]\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history")
async def get_chat_history() -> HistoryResponse:
    """Load all display messages across all sessions."""
    return HistoryResponse(messages=get_messages())


@router.post("/new-session")
async def new_chat_session() -> SessionResponse:
    """Start a new global conversation session."""
    return SessionResponse(session_id=start_new_session())


@router.delete("/history")
async def clear_chat_history() -> dict:
    """Clear all chat history and reset the agent's in-memory state."""
    delete_messages()
    reset_agent()
    return {"status": "ok"}


@router.delete("/session/{session_id}")
async def delete_chat_session(session_id: int) -> dict:
    """Delete a single conversation session and reset the agent."""
    delete_session(session_id)
    reset_agent()
    return {"status": "ok", "session_id": session_id}
