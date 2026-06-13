# tests/test_compact.py
"""Standalone test for checkpoint compaction — no UI/HTTP needed.

Drives the agent directly through a few chat turns on one global session
(thread_id = "session-{session_id}"), then inspects and triggers
compact_checkpointer_if_needed.

Run:
    python tests/test_compact.py
"""
import asyncio
import sys
import time

sys.path.insert(0, ".")

from langchain_core.messages import AIMessage

from backend.core.config import get_settings
from backend.core.database import init_db, get_db
from backend.services.agent import get_document_agent, set_tool_context
from backend.services.compactor import (
    compact_checkpointer_if_needed,
    _estimate_bmessage_tokens,
)

# Each turn (esp. with retrieve_context tool calls + summarization calls)
# can cost several thousand tokens. Groq oss-120b free tier limit is
# 8K tokens/minute, so space out turns to stay under it.
DELAY_BETWEEN_CALLS_SEC = 65

# Use a real doc_id from your vault (any document with content/tree_index).
DOC_ID = "9f34b22e-8329-4f52-ad13-9bad122660d8"
SESSION_ID = 999999  # dedicated test session, won't collide with real sessions
THREAD_ID = f"session-{SESSION_ID}"


def _doc_titles(doc_ids: list[str]) -> dict[str, str]:
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


def describe_messages(messages, label: str):
    tokens = _estimate_bmessage_tokens(messages)
    print(f"\n{'='*20} {label} {'='*20}")
    print(f"Total: {len(messages)} messages | ~{tokens} tokens (estimate)")
    for i, m in enumerate(messages):
        content = getattr(m, "content", "")
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        preview = (content or "")[:80].replace("\n", " ")
        extra = ""
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            extra = f" [tool_calls={[tc.get('name') for tc in m.tool_calls]}]"
        print(f"  [{i}] {type(m).__name__:<14} {preview}{extra}")


async def ask(agent, question: str, config: dict):
    """Send one turn through the agent (consumes the stream fully)."""
    full = ""
    async for chunk in agent.astream_events(
        {"messages": [{"role": "user", "content": question}]},
        version="v2",
        config=config,
    ):
        if chunk.get("event") != "on_chat_model_stream":
            continue
        if chunk.get("metadata", {}).get("langgraph_node") == "tools":
            continue
        msg = chunk.get("data", {}).get("chunk")
        if not msg:
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        full += content or ""
    print(f"\n  Q: {question}")
    print(f"  A: {full[:200]}")
    return full


async def main():
    init_db()
    settings = get_settings()

    agent = await get_document_agent()

    doc_ids = [DOC_ID]
    titles = _doc_titles(doc_ids)
    set_tool_context(doc_ids, settings, titles)

    config = {"configurable": {"thread_id": THREAD_ID}}

    def wait(label: str):
        print(f"\n[Waiting {DELAY_BETWEEN_CALLS_SEC}s before {label} to respect rate limit...]")
        time.sleep(DELAY_BETWEEN_CALLS_SEC)

    # ---- Drive a few turns so the checkpoint accumulates messages -------
    print("Sending 3 turns to build up checkpoint state...")
    await ask(agent, "Lịch sử phát triển AI bắt đầu từ thập niên nào?", config)

    wait("turn 2")
    await ask(agent, "Mùa đông AI gây ra do gì?", config)

    wait("turn 3")
    await ask(agent, "AI bùng nổ trở lại nhờ những yếu tố nào?", config)

    # ---- Inspect checkpoint BEFORE compaction ----------------------------
    tup = await agent.checkpointer.aget_tuple(config)
    messages_before = tup.checkpoint["channel_values"].get("messages", [])
    describe_messages(messages_before, "CHECKPOINT BEFORE COMPACT")

    # ---- Force compaction with a low threshold ---------------------------
    # Use a threshold lower than current token count so it definitely triggers.
    current_tokens = _estimate_bmessage_tokens(messages_before)
    test_threshold = max(50, current_tokens // 2)

    wait("compaction call")
    print(f"\nRunning compact_checkpointer_if_needed(threshold={test_threshold})...")
    result = await compact_checkpointer_if_needed(
        agent, THREAD_ID, settings, threshold=test_threshold, tail_pairs=1
    )
    print(f"Compacted: {result}")

    # ---- Inspect checkpoint AFTER compaction ------------------------------
    tup2 = await agent.checkpointer.aget_tuple(config)
    messages_after = tup2.checkpoint["channel_values"].get("messages", [])
    describe_messages(messages_after, "CHECKPOINT AFTER COMPACT")

    # ---- Sanity check: agent still works with compacted state ------------
    if result:
        wait("follow-up question")
        print("\nAsking a follow-up question with compacted context...")
        await ask(agent, "Tôi đã hỏi câu đầu tiên về điều gì?", config)

    # ---- Cleanup: remove the dedicated test thread from checkpoints.db ---
    try:
        await agent.checkpointer.adelete_thread(THREAD_ID)
        print(f"\nCleaned up test thread '{THREAD_ID}' from checkpoints.db")
    except AttributeError:
        print(
            f"\nNote: checkpointer has no adelete_thread; "
            f"test thread '{THREAD_ID}' left in checkpoints.db"
        )
    except Exception as exc:
        print(f"\nCleanup warning: {exc}")


if __name__ == "__main__":
    asyncio.run(main())