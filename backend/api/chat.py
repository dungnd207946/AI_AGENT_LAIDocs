"""Chat API — multi-document Q&A via a LangGraph ReAct agent.

Endpoints:
  POST   /api/chat/stream              - Stream answer over selected doc scope (SSE)
  GET    /api/chat/history             - Load all display messages (all sessions)
  POST   /api/chat/new-session         - Start a fresh global session
  DELETE /api/chat/history             - Clear all history
  DELETE /api/chat/session/{session_id} - Delete one session
"""

from __future__ import annotations

import hashlib
import json
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
    get_reasoning_chain,
)
from ..services.chat_history import (
    get_current_session_id,
    get_display_messages,
    save_message,
    save_message_evidence,
    save_message_chain,
    start_new_session,
    delete_messages,
    delete_session,
)
from ..services.compactor import compact_checkpointer_if_needed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    doc_ids: list[str]
    question: str
    session_id: int | None = None  # If None, use current global session


class CompareRequest(BaseModel):
    doc_id: str
    question: str


class CompareArm(BaseModel):
    answer: str
    units: list[dict]  # [{unit_id, title, heading_path, preview, kind}]


class CompareResponse(BaseModel):
    doc_id: str
    question: str
    rag: CompareArm
    graph: CompareArm
    bridge_unit_ids: list[str]  # units GraphRAG surfaced that plain RAG missed


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
        agent = None
        config = None
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
            evidence = get_retrieved_evidence()
            chain = get_reasoning_chain()
            # Sessions are global; scope is transient, so evidence/chain are
            # tagged with the first in-scope doc as a representative doc_id.
            doc_id = body.doc_ids[0] if body.doc_ids else ""
            if full_response:
                try:
                    save_message(session_id, "user", body.question)
                    assistant_message_id = save_message(session_id, "assistant", full_response)
                    save_message_evidence(assistant_message_id, doc_id, evidence)
                    save_message_chain(assistant_message_id, doc_id, chain)
                except Exception:
                    logger.exception("Failed to save chat history")
            # Emit citation evidence so the UI can render source chips + jump-to-source.
            if evidence:
                try:
                    payload = json.dumps(evidence, ensure_ascii=False)
                    yield f"data: [EVIDENCE] {payload}\n\n"
                except Exception:
                    logger.exception("Failed to serialize evidence")
            # Emit the graph-of-thought reasoning chain if one was produced.
            if chain:
                yield f"data: [CHAIN] {json.dumps(chain, ensure_ascii=False)}\n\n"
            # Signal the frontend to reload the document if the agent edited it
            if document_was_edited():
                yield "data: [EDITED]\n\n"

            # Compact the LangGraph checkpoint (the agent's durable working
            # memory) so the NEXT turn in this session starts from a smaller
            # message list. pre_model_hook only trims what the LLM sees per
            # request — it doesn't shrink the persisted checkpoint, so old
            # turns would otherwise accumulate (and eventually be hard-dropped
            # with no summary). This never touches chat_messages/display
            # history or evidence/chain persistence above.
            if agent is not None and config is not None:
                try:
                    await compact_checkpointer_if_needed(
                        agent, config["configurable"]["thread_id"], settings
                    )
                except Exception:
                    logger.exception("compact_checkpointer_if_needed failed; continuing")

            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history")
async def get_chat_history() -> HistoryResponse:
    """Load all display messages across all sessions.

    Assistant messages carry their citation ``evidence`` and graph-of-thought
    ``chain`` so the UI can re-render source chips and reasoning paths on reload.
    """
    return HistoryResponse(messages=get_display_messages())


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


_COMPARE_SYSTEM_PROMPT = (
    "You are a document assistant. Answer the user's question using ONLY the "
    "provided context. If the answer is not present in the context, reply "
    "exactly: \"I don't see this in the document.\" Be concise; cite the section "
    "title you used. Do not use outside knowledge."
)


def _arm_units(unit_ids: list[str], unit_map: dict) -> list[dict]:
    """Project ranked unit ids onto display metadata for a compare arm."""
    from ..services import retrieval as r

    out: list[dict] = []
    for uid in unit_ids:
        unit = unit_map.get(uid)
        if not unit:
            continue
        out.append(
            {
                "unit_id": uid,
                "title": unit.get("title") or "",
                "kind": unit.get("kind") or "text",
                "heading_path": r._evidence_heading_path(unit),
                "preview": r._evidence_preview(unit),
            }
        )
    return out


@router.post("/compare")
async def compare_retrieval(body: CompareRequest) -> CompareResponse:
    """Answer the same question twice — plain RAG vs GraphRAG — for the demo.

    Stateless (no session/history): runs ``hybrid_rank`` with the graph signal
    OFF then ON, builds context from each ranked unit set, and asks the LLM both
    times with an identical grounded prompt. The only variable is whether the
    entity-relation graph was walked, so the answers (and the extra "bridge"
    unit GraphRAG recovers) isolate the GraphRAG contribution.
    """
    settings = get_settings()
    from ..services.llm import is_llm_configured, create_chat_model
    if not is_llm_configured(settings.active_llm):
        raise HTTPException(
            status_code=503,
            detail="LLM is not configured. Please set the LLM endpoint in Settings.",
        )

    from ..services import retrieval as r

    units = r.get_retrieval_units(body.doc_id)
    if not units:
        raise HTTPException(status_code=404, detail="Document has no retrievable content.")
    unit_map = {str(u.get("unit_id")): u for u in units if u.get("unit_id") is not None}

    def _rank(graph_enabled: bool) -> list[str]:
        cfg = settings.model_copy(deep=True)
        cfg.graph_rag.enabled = graph_enabled
        fused, _tree = r.hybrid_rank(body.doc_id, body.question, cfg, units=units)
        return fused

    model = create_chat_model(settings.active_llm)

    def _answer(unit_ids: list[str]) -> str:
        selected = [unit_map[uid] for uid in unit_ids if uid in unit_map]
        if not selected:
            return "I don't see this in the document."
        context = r.build_context_from_units(selected)
        try:
            resp = model.invoke(
                [
                    {"role": "system", "content": _COMPARE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {body.question}"},
                ]
            )
        except Exception as exc:
            logger.exception("Compare LLM call failed")
            return f"[error generating answer: {exc}]"
        content = resp.content
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return (content or "").strip() or "I don't see this in the document."

    rag_ids = _rank(False)
    graph_ids = _rank(True)
    bridge = [uid for uid in graph_ids if uid not in set(rag_ids)]

    return CompareResponse(
        doc_id=body.doc_id,
        question=body.question,
        rag=CompareArm(answer=_answer(rag_ids), units=_arm_units(rag_ids, unit_map)),
        graph=CompareArm(answer=_answer(graph_ids), units=_arm_units(graph_ids, unit_map)),
        bridge_unit_ids=bridge,
    )


@router.delete("/session/{session_id}")
async def delete_chat_session(session_id: int) -> dict:
    """Delete a single conversation session and reset the agent.

    Resets the agent so the deleted session's in-memory state is discarded.
    """
    delete_session(session_id)
    reset_agent()
    return {"status": "ok", "session_id": session_id}