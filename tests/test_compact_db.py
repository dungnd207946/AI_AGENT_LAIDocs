"""Tests for the checkpoint compactor (backend/services/compactor.py).

These exercise `compact_checkpointer_if_needed`, which bounds the LangGraph
conversation checkpoint by replacing older turns with an LLM-generated summary
while keeping the last few Human/AI exchanges verbatim.

Everything is in-memory and deterministic: a `MemorySaver` stands in for the
real AsyncSqliteSaver, and the summarizer LLM is faked, so no real DB, network,
or model is touched.

(Replaces an earlier manual script that imported the now-removed
`compact_if_needed` — the display-history compactor superseded by the
checkpoint-based one in the checkpoint-compaction refactor.)
"""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import MemorySaver

from backend.services import compactor


THREAD_ID = "session-1"
_CONFIG = {"configurable": {"thread_id": THREAD_ID, "checkpoint_ns": ""}}


class _FakeModel:
    """Stand-in for the summarizer LLM; records that it was called."""

    def __init__(self, summary: str = "- user asked about X\n- doc says Y"):
        self.summary = summary
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.summary)


def _fake_settings():
    # create_chat_model is patched, so active_llm is never really consumed.
    return SimpleNamespace(active_llm=object())


def _agent_with(messages):
    """Return a fake agent whose MemorySaver holds a seeded checkpoint."""
    saver = MemorySaver()
    ckpt = empty_checkpoint()
    ckpt["channel_values"]["messages"] = messages
    ckpt["channel_versions"]["messages"] = "1"
    asyncio.run(saver.aput(_CONFIG, ckpt, {}, {"messages": "1"}))
    return SimpleNamespace(checkpointer=saver)


def _stored_messages(saver) -> list:
    tup = asyncio.run(saver.aget_tuple(_CONFIG))
    return tup.checkpoint["channel_values"]["messages"]


def _long_conversation():
    """Three Human/AI turns; the first turn is long enough to blow a small
    token threshold once measured across the whole list."""
    return [
        HumanMessage(content="Q1 " + "detail " * 40),
        AIMessage(content="A1 " + "answer " * 40),
        HumanMessage(content="Q2 short"),
        AIMessage(content="A2 short"),
        HumanMessage(content="Q3 short"),
        AIMessage(content="A3 short"),
    ]


def test_compacts_when_over_threshold(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr("backend.services.llm.create_chat_model", lambda cfg: fake)

    messages = _long_conversation()
    agent = _agent_with(messages)

    did = asyncio.run(
        compactor.compact_checkpointer_if_needed(
            agent, THREAD_ID, _fake_settings(), threshold=20, tail_pairs=2
        )
    )

    assert did is True
    assert fake.calls == 1
    new_messages = _stored_messages(agent.checkpointer)
    # Original 6 messages -> summary + last 2 user turns (Q2/A2/Q3/A3) = 5.
    assert len(new_messages) == 5
    assert isinstance(new_messages[0], AIMessage)
    assert new_messages[0].content.startswith("[Earlier conversation summary]")
    assert fake.summary in new_messages[0].content
    # The verbatim tail is preserved untouched.
    assert [m.content for m in new_messages[1:]] == [
        "Q2 short",
        "A2 short",
        "Q3 short",
        "A3 short",
    ]


def test_noop_when_under_threshold(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr("backend.services.llm.create_chat_model", lambda cfg: fake)

    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    agent = _agent_with(messages)

    did = asyncio.run(
        compactor.compact_checkpointer_if_needed(
            agent, THREAD_ID, _fake_settings(), threshold=10_000, tail_pairs=2
        )
    )

    assert did is False
    assert fake.calls == 0  # LLM never invoked
    assert len(_stored_messages(agent.checkpointer)) == 2


def test_noop_when_too_few_turns_to_split(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr("backend.services.llm.create_chat_model", lambda cfg: fake)

    # Over threshold, but only 2 user turns with tail_pairs=2 -> nothing to compact.
    messages = [
        HumanMessage(content="Q1 " + "x " * 100),
        AIMessage(content="A1 " + "y " * 100),
        HumanMessage(content="Q2 " + "x " * 100),
        AIMessage(content="A2 " + "y " * 100),
    ]
    agent = _agent_with(messages)

    did = asyncio.run(
        compactor.compact_checkpointer_if_needed(
            agent, THREAD_ID, _fake_settings(), threshold=20, tail_pairs=2
        )
    )

    assert did is False
    assert fake.calls == 0
    assert len(_stored_messages(agent.checkpointer)) == 4


def test_noop_when_no_checkpointer():
    agent = SimpleNamespace(checkpointer=None)
    did = asyncio.run(
        compactor.compact_checkpointer_if_needed(agent, THREAD_ID, _fake_settings())
    )
    assert did is False


def test_noop_when_summarizer_returns_empty(monkeypatch):
    """A blank summary must not clobber the stored conversation."""
    fake = _FakeModel(summary="   ")
    monkeypatch.setattr("backend.services.llm.create_chat_model", lambda cfg: fake)

    messages = _long_conversation()
    agent = _agent_with(messages)

    did = asyncio.run(
        compactor.compact_checkpointer_if_needed(
            agent, THREAD_ID, _fake_settings(), threshold=20, tail_pairs=2
        )
    )

    assert did is False
    assert len(_stored_messages(agent.checkpointer)) == 6  # unchanged
