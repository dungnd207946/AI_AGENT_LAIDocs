"""Conversation compactor — rolling summary to reduce token usage.

Strategy:
  - Estimate tokens from display history (chat_messages table).
  - If total > threshold: compact everything except the last TAIL_PAIRS Q&A pairs
    into a single summary via LLM, store as a special summary row.
  - Next request loads: [summary_message] + [tail messages] instead of full history.

Compact chain:
  Chat1, Chat2 → Summary_1
  Summary_1, Chat3, Chat4 → Summary_2   (tail=2 pairs kept raw, rest compacted)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COMPACT_THRESHOLD_TOKENS = 500 # trigger compaction above this estimate
TAIL_PAIRS = 2                   # Q&A pairs to keep verbatim (not compacted)
TAIL_MESSAGES = TAIL_PAIRS * 2   # = 4 messages

_COMPACT_SYSTEM = (
    "You are a conversation summarizer. "
    "Produce a compact bullet-point summary that preserves: "
    "key facts found in the document, questions asked, answers given, "
    "and any edits made. Be concise. No preamble, no conclusion sentence."
)

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(messages: list[dict]) -> int:
    """Rough estimate: 1 token ≈ 4 chars of content."""
    return sum(len(m.get("content", "")) for m in messages) // 4


# ---------------------------------------------------------------------------
# LLM call for compaction
# ---------------------------------------------------------------------------


async def _call_llm_compact(history_text: str, settings: "Settings") -> str:
    """Call the configured LLM to summarize history_text."""
    from ..services.llm import create_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage

    model = create_chat_model(settings.active_llm)
    response = await model.ainvoke([
        SystemMessage(content=_COMPACT_SYSTEM),
        HumanMessage(content=f"Summarize this conversation:\n\n{history_text}"),
    ])
    content = response.content
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return content.strip()


def _format_for_compact(messages: list[dict]) -> str:
    """Format messages list into readable text for the LLM."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "unknown").upper()
        content = m.get("content", "")
        # Truncate very long assistant responses to first 800 chars to save tokens
        if role == "ASSISTANT" and len(content) > 800:
            content = content[:800] + "… [truncated]"
        lines.append(f"[{role}]: {content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def compact_if_needed(
    session_id: int,
    settings: "Settings",
    threshold: int = COMPACT_THRESHOLD_TOKENS,
) -> bool:
    """Check a session's display history; compact if over threshold.

    Returns True if compaction was performed. Reads/writes via chat_history.
    """
    from ..services.chat_history import (
        get_messages_for_compact,
        save_compact_summary,
        delete_compacted_messages,
    )

    messages = get_messages_for_compact(session_id)

    if len(messages) <= TAIL_MESSAGES:
        return False

    total_tokens = estimate_tokens(messages)
    if total_tokens <= threshold:
        return False

    body = messages[:-TAIL_MESSAGES]
    tail = messages[-TAIL_MESSAGES:]

    if estimate_tokens(tail) >= threshold:
        body = messages[:-2]
        tail = messages[-2:]

    logger.info(
        "Compacting %d messages (est. %d tokens) for session %s",
        len(body), total_tokens, session_id,
    )

    history_text = _format_for_compact(body)
    try:
        summary = await _call_llm_compact(history_text, settings)
    except Exception:
        logger.exception("Compaction LLM call failed; skipping compact")
        return False

    body_ids = [m["id"] for m in body]
    save_compact_summary(session_id, summary)
    delete_compacted_messages(body_ids)

    logger.info("Compaction done for session %s; summary saved", session_id)
    return True