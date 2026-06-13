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
from langgraph.types import Command
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


class ResumeRequest(BaseModel):
    """Resume a turn paused at an edit-confirmation gate (apply_edit interrupt)."""
    doc_ids: list[str]
    decision: str  # "approve" | "reject"
    session_id: int | None = None


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


async def _has_pending_interrupt(agent, config) -> bool:
    """True if this thread is suspended at an interrupt() gate (awaiting resume)."""
    try:
        snapshot = await agent.aget_state(config)
        return bool(snapshot.interrupts)
    except Exception:
        logger.exception("Failed to read agent state for pending-interrupt check")
        return False


async def _clear_pending_interrupt(agent, config) -> None:
    """Discard a stale edit-confirmation gate the user walked away from.

    A pending interrupt wedges the thread: feeding a *new* question while one is
    suspended leaves the interrupt in place (LangGraph expects Command(resume=…)).
    So before a fresh turn we resume with "reject" and drain it — the edit is
    never applied, and the thread is left clean for the new question. Silent
    (tokens are not streamed); only the checkpointer is advanced.
    """
    try:
        async for _ in agent.astream_events(
            Command(resume="reject"), version="v2", config=config
        ):
            pass
    except Exception:
        logger.exception("Failed to clear stale interrupt; proceeding anyway")


def _build_chat_config(session_id: int, doc_ids: list[str], run_name: str) -> dict:
    """Run config for one agent turn. thread_id is per global session → the
    AsyncSqliteSaver replays the full conversation (and any pending interrupt)
    on resume, independent of doc scope."""
    return {
        "configurable": {"thread_id": f"session-{session_id}"},
        "run_name": run_name,
        "metadata": {"doc_ids": doc_ids, "session_id": session_id},
        "tags": ["ai-agent-chatbot"],
    }


async def _run_turn(agent, stream_input, config, session_id, settings, doc_ids, user_question):
    """Stream one agent turn (a fresh question OR a Command(resume=…)) as SSE.

    Emits AI content tokens, then a trailer of sentinels: ``[EVIDENCE]`` /
    ``[CHAIN]`` always, then EITHER ``[INTERRUPT] {payload}`` when the turn
    paused at the edit-confirmation gate, OR ``[EDITED]`` (+ checkpoint
    compaction) when it ran to completion. Always closes with ``[DONE]``.
    """
    full_response = ""
    try:
        async for chunk in agent.astream_events(stream_input, version="v2", config=config):
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
        logger.exception("Chat turn error")
        yield f"data: [ERROR] {exc}\n\n"
    finally:
        # Did the turn pause at an interrupt() gate (apply_edit awaiting approval)?
        interrupted = False
        interrupt_payload = None
        try:
            snapshot = await agent.aget_state(config)
            if snapshot.interrupts:
                interrupted = True
                interrupt_payload = snapshot.interrupts[0].value
        except Exception:
            logger.exception("Failed to read agent state for interrupt detection")

        evidence = get_retrieved_evidence()
        chain = get_reasoning_chain()
        # Sessions are global; scope is transient, so evidence/chain are tagged
        # with the first in-scope doc as a representative doc_id.
        doc_id = doc_ids[0] if doc_ids else ""
        try:
            # Save the user question once (only the originating /stream turn
            # carries one; a /resume turn passes user_question=None).
            if user_question and (full_response or interrupted):
                save_message(session_id, "user", user_question)
            if full_response:
                assistant_message_id = save_message(session_id, "assistant", full_response)
                save_message_evidence(assistant_message_id, doc_id, evidence)
                save_message_chain(assistant_message_id, doc_id, chain)
        except Exception:
            logger.exception("Failed to save chat history")

        if evidence:
            try:
                yield f"data: [EVIDENCE] {json.dumps(evidence, ensure_ascii=False)}\n\n"
            except Exception:
                logger.exception("Failed to serialize evidence")
        if chain:
            yield f"data: [CHAIN] {json.dumps(chain, ensure_ascii=False)}\n\n"

        if interrupted:
            # Paused mid-turn: hand the diff to the UI for approve/reject. The
            # edit is NOT applied yet, and we must NOT compact a thread that has
            # a pending task (it would drop the suspended tool call).
            try:
                yield f"data: [INTERRUPT] {json.dumps(interrupt_payload, ensure_ascii=False)}\n\n"
            except Exception:
                logger.exception("Failed to serialize interrupt payload")
        else:
            # Signal the frontend to reload the document if the agent edited it.
            if document_was_edited():
                yield "data: [EDITED]\n\n"
            # Compact the LangGraph checkpoint so the NEXT turn starts from a
            # smaller message list. Never touches chat_messages/display history.
            try:
                await compact_checkpointer_if_needed(
                    agent, config["configurable"]["thread_id"], settings
                )
            except Exception:
                logger.exception("compact_checkpointer_if_needed failed; continuing")

        yield "data: [DONE]\n\n"


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
        try:
            agent = await get_document_agent()
        except Exception as exc:
            logger.exception("Failed to build agent")
            yield f"data: [ERROR] {exc}\n\n"
            yield "data: [DONE]\n\n"
            return
        config = _build_chat_config(session_id, body.doc_ids, "document-chat")
        # If a previous turn was left at the edit-confirmation gate and the user
        # moved on with a new question, clear that stale interrupt first (the
        # abandoned edit is safely discarded) so the thread isn't wedged.
        if await _has_pending_interrupt(agent, config):
            logger.info("Discarding stale edit interrupt before new question (session %s)", session_id)
            await _clear_pending_interrupt(agent, config)
        # Only the new question is sent; prior turns live in the checkpointer
        # (no manual prior-injection → no double-feeding the conversation).
        stream_input = {"messages": [{"role": "user", "content": body.question}]}
        async for event in _run_turn(
            agent, stream_input, config, session_id, settings, body.doc_ids, body.question
        ):
            yield event

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/resume")
async def chat_resume(body: ResumeRequest):
    """Resume a turn paused at the edit-confirmation gate (SSE stream).

    The agent's ``apply_edit`` tool called ``interrupt()`` and the graph is
    suspended in the checkpointer. We feed back ``Command(resume=decision)`` on
    the SAME thread_id: ``apply_edit`` re-runs, applies the edit on "approve"
    (or leaves the file untouched on "reject"), and the agent finishes its turn.
    """
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
    # apply_edit re-executes on resume and reads this context to resolve the
    # target file and to flag the edit — it must be set before streaming.
    set_tool_context(body.doc_ids, settings, titles)
    decision = "approve" if body.decision.strip().lower() == "approve" else "reject"

    async def _event_generator():
        try:
            agent = await get_document_agent()
        except Exception as exc:
            logger.exception("Failed to build agent")
            yield f"data: [ERROR] {exc}\n\n"
            yield "data: [DONE]\n\n"
            return
        config = _build_chat_config(session_id, body.doc_ids, "document-chat-resume")
        # Guard a stale/duplicate resume (e.g. double-click, or the gate was
        # already cleared): resuming a thread with no pending interrupt would
        # mis-run the graph. Nothing to do → close the stream cleanly.
        if not await _has_pending_interrupt(agent, config):
            logger.info("Resume with no pending interrupt — no-op (session %s)", session_id)
            yield "data: [DONE]\n\n"
            return
        async for event in _run_turn(
            agent, Command(resume=decision), config, session_id, settings, body.doc_ids, None
        ):
            yield event

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

    Stateless (no session/history). The RAG arm is ``hybrid_rank`` with the
    graph signal OFF. The GraphRAG arm takes that same base set and *adds* the
    bridge units the entity-relation walk recovered — units that RRF fusion +
    the top-k clip would otherwise drown out, which is why the two arms used to
    return near-identical answers. Both arms run the LLM with an identical
    grounded prompt, so any divergence isolates exactly what the graph walk
    contributed (and ``bridge_unit_ids`` names those units for the UI). When the
    graph recovers nothing for the question, the arms are intentionally identical.
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

    def _graph_only_units(base_ids: list[str]) -> list[str]:
        """Units the graph walk recovers that the base RAG ranking missed.

        ``hybrid_rank`` folds the graph signal into RRF and then clips to the
        fused top-k, so graph-only "bridge" units (a single weak vote) almost
        never survive — making the two arms look identical. Pull them straight
        from ``graph_search`` instead, capped by ``graph_rag.max_units``.
        """
        graph_cfg = settings.model_copy(deep=True)
        graph_cfg.graph_rag.enabled = True
        try:
            recovered = r.graph_search(body.doc_id, body.question, graph_cfg, units=units)
        except Exception:
            logger.exception("Graph search failed in compare for doc %s", body.doc_id)
            return []
        base = set(base_ids)
        cap = max(0, graph_cfg.graph_rag.max_units)
        return [uid for uid in recovered if uid in unit_map and uid not in base][:cap]

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
    bridge = _graph_only_units(rag_ids)
    # GraphRAG arm = base set + the connected passages only the graph walk found,
    # so the extra context can actually change the answer (and be cited).
    graph_ids = rag_ids + bridge

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