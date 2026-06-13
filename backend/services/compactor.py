"""Checkpoint compactor — bounds the LangGraph conversation checkpoint.

Context:
  - Sessions are global; the agent's full memory lives in the AsyncSqliteSaver
    checkpoint for thread_id "session-{session_id}" (~/.laidocs/data/checkpoints.db).
  - `pre_model_hook` (_trim_to_recent_turns in agent.py) only bounds what the
    LLM SEES this step (`llm_input_messages`) — it does NOT shrink the
    persisted checkpoint. Over a long session, `channel_values["messages"]`
    grows unboundedly (including large ToolMessage payloads from
    `retrieve_context` / `reason_over_graph`), which:
      - bloats checkpoints.db on disk
      - means turns trimmed by pre_model_hook are HARD-DROPPED from the
        model's view forever, with no summary of what they contained.

`compact_checkpointer_if_needed` fixes both: if the stored message list is
over a token threshold, it replaces everything except the last
CHECKPOINT_TAIL_PAIRS Human/AI exchanges with a single LLM-generated summary
AIMessage, then writes the smaller list back via the checkpointer's
aget_tuple/aput — the same Checkpoint structure regardless of storage backend
(MemorySaver, AsyncSqliteSaver, ...), so this requires no agent.py changes.

This is best-effort: it touches internal checkpoint fields (`channel_values`,
`channel_versions`) that are not part of the public LangGraph API and may
differ across versions. Any failure is logged and skipped — it never raises,
and never touches `chat_messages` / display history.
"""

from __future__ import annotations

import copy
import logging
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_COMPACT_THRESHOLD_TOKENS = 4000
CHECKPOINT_TAIL_PAIRS = 2  # Human/AI exchanges kept verbatim (incl. their tool calls)

_CHECKPOINT_COMPACT_SYSTEM = (
    "You are a conversation summarizer for an AI agent's working memory. "
    "Summarize the exchanges below into compact bullet points, preserving: "
    "questions the user asked, key facts/answers retrieved from the documents "
    "(including specific file names, section names, figures, numbers), and any "
    "edits made. Be concise. No preamble, no conclusion sentence."
)


def _new_checkpoint_id() -> str:
    """Generate a checkpoint id LangGraph will treat as the latest.

    AsyncSqliteSaver picks the "latest" checkpoint for a thread via
    ``ORDER BY checkpoint_id DESC``. LangGraph generates checkpoint ids as
    UUID6 (time-ordered, lexicographically sortable) — a random uuid4 can
    sort BEFORE an existing UUID6 id, so the compacted checkpoint would be
    written but never picked up as "latest". Use the same uuid6 generator
    LangGraph uses internally; fall back to uuid1 (also time-based) if the
    helper isn't importable in this langgraph-checkpoint version.
    """
    try:
        from langgraph.checkpoint.base.id import uuid6
        return str(uuid6())
    except Exception:
        logger.warning("uuid6 helper unavailable; falling back to uuid1")
        return str(uuid.uuid1())


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _bmessage_text(msg) -> str:
    """Extract plain text content from a BaseMessage (handles list-content)."""
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return content or ""


def _estimate_bmessage_tokens(messages: list) -> int:
    """Rough estimate: 1 token ≈ 4 chars of content."""
    return sum(len(_bmessage_text(m)) for m in messages) // 4


def _format_bmessages_for_compact(messages: list) -> str:
    """Format a list of BaseMessage (Human/AI/Tool) into text for the LLM."""
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    lines: list[str] = []
    for m in messages:
        text = _bmessage_text(m)
        if isinstance(m, HumanMessage):
            role = "USER"
        elif isinstance(m, AIMessage):
            role = "ASSISTANT"
            if not text and getattr(m, "tool_calls", None):
                calls = ", ".join(tc.get("name", "?") for tc in m.tool_calls)
                text = f"[called tool(s): {calls}]"
        elif isinstance(m, ToolMessage):
            role = "TOOL_RESULT"
            if len(text) > 1000:
                text = text[:1000] + "… [truncated]"
        else:
            role = type(m).__name__.upper()
        if text:
            lines.append(f"[{role}]: {text}")
    return "\n\n".join(lines)


def _split_tail_at_human_boundary(messages: list, tail_pairs: int) -> tuple[list, list]:
    """Split messages into (body, tail), where tail starts at a HumanMessage
    boundary and contains roughly the last `tail_pairs` user turns
    (including any tool-call/tool-result messages that followed them).
    """
    from langchain_core.messages import HumanMessage

    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_indices) <= tail_pairs:
        return [], messages

    split_at = human_indices[-tail_pairs]
    return messages[:split_at], messages[split_at:]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def compact_checkpointer_if_needed(
    agent,
    thread_id: str,
    settings: "Settings",
    threshold: int = CHECKPOINT_COMPACT_THRESHOLD_TOKENS,
    tail_pairs: int = CHECKPOINT_TAIL_PAIRS,
) -> bool:
    """Compact the LangGraph checkpoint for `thread_id` in-place.

    Call this AFTER a turn completes (e.g. in the `finally` block of the
    stream generator) so the NEXT turn starts from a smaller message list.
    Works with any checkpointer backend (MemorySaver, AsyncSqliteSaver, ...)
    since it only relies on the standard Checkpoint structure.

    Returns True if compaction was performed.
    """
    from langchain_core.messages import AIMessage

    checkpointer = agent.checkpointer
    if checkpointer is None:
        return False

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    try:
        tup = await checkpointer.aget_tuple(config)
    except Exception:
        logger.exception("compact_checkpointer_if_needed: aget_tuple failed")
        return False

    if tup is None:
        return False

    checkpoint = tup.checkpoint
    channel_values = checkpoint.get("channel_values", {})
    messages = channel_values.get("messages")
    if not messages:
        return False

    total_tokens = _estimate_bmessage_tokens(messages)
    if total_tokens <= threshold:
        return False

    body, tail = _split_tail_at_human_boundary(messages, tail_pairs)
    if not body:
        # Not enough turns to compact below the tail boundary.
        return False

    logger.info(
        "Compacting checkpoint for thread %s: %d/%d messages (~%d tokens)",
        thread_id, len(body), len(messages), total_tokens,
    )

    history_text = _format_bmessages_for_compact(body)
    try:
        from ..services.llm import create_chat_model
        from langchain_core.messages import HumanMessage, SystemMessage

        model = create_chat_model(settings.active_llm)
        response = await model.ainvoke([
            SystemMessage(content=_CHECKPOINT_COMPACT_SYSTEM),
            HumanMessage(content=f"Summarize this conversation:\n\n{history_text}"),
        ])
        summary = _bmessage_text(response).strip()
        if not summary:
            return False
    except Exception:
        logger.exception("compact_checkpointer_if_needed: LLM summarization failed")
        return False

    summary_msg = AIMessage(content=f"[Earlier conversation summary]\n{summary}")
    new_messages = [summary_msg, *tail]

    # ---- Write the compacted state back to the checkpointer -------------
    try:
        new_checkpoint = copy.deepcopy(checkpoint)
        new_checkpoint["id"] = _new_checkpoint_id()
        new_checkpoint["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S.000000+00:00", time.gmtime())
        new_checkpoint["channel_values"]["messages"] = new_messages

        old_versions = checkpoint.get("channel_versions", {})
        old_msg_version = old_versions.get("messages")
        try:
            new_msg_version = checkpointer.get_next_version(old_msg_version, None)
        except Exception:
            # Fallback for checkpointer implementations without get_next_version
            new_msg_version = (old_msg_version or 0) + 1  # type: ignore[operator]

        new_checkpoint["channel_versions"] = {
            **old_versions,
            "messages": new_msg_version,
        }

        # AsyncSqliteSaver.aput requires config["configurable"]["checkpoint_ns"]
        # (and uses checkpoint_id to link the new checkpoint as the next one
        # in this thread). `tup.config` is the full config returned by
        # aget_tuple for the checkpoint we just read — reuse it as the parent
        # config rather than the minimal {thread_id} dict we built above.
        await checkpointer.aput(
            tup.config,
            new_checkpoint,
            tup.metadata or {},
            {"messages": new_msg_version},
        )
    except Exception:
        logger.exception("compact_checkpointer_if_needed: aput failed; state unchanged")
        return False

    logger.info(
        "Checkpoint compacted for thread %s: %d -> %d messages",
        thread_id, len(messages), len(new_messages),
    )
    return True