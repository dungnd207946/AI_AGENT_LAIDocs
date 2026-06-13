"""Tests for the conversation-window trimming hook (Hướng A: checkpointer is
the agent's memory; we cap what the model sees to the last N user turns)."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.services.agent import _trim_to_recent_turns, MAX_RECENT_TURNS


def _turn(i: int) -> list:
    """One plain Q&A turn: user question + assistant answer."""
    return [HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")]


def test_keeps_all_when_under_limit():
    msgs = [m for i in range(MAX_RECENT_TURNS - 1) for m in _turn(i)]
    out = _trim_to_recent_turns({"messages": msgs})["llm_input_messages"]
    assert out == msgs  # nothing dropped


def test_trims_to_last_n_turns():
    total = MAX_RECENT_TURNS + 3
    msgs = [m for i in range(total) for m in _turn(i)]
    out = _trim_to_recent_turns({"messages": msgs})["llm_input_messages"]
    humans = [m for m in out if isinstance(m, HumanMessage)]
    assert len(humans) == MAX_RECENT_TURNS
    # The window must start exactly at the first kept user turn.
    assert isinstance(out[0], HumanMessage)
    assert out[0].content == f"q{total - MAX_RECENT_TURNS}"


def test_window_starts_on_human_so_tool_pairs_are_never_orphaned():
    """A turn that used a tool must not be cut mid-sequence: trimming starts
    on a HumanMessage, so an AI tool-call is always preceded by its user turn
    and followed by its ToolMessage."""
    msgs: list = []
    for i in range(MAX_RECENT_TURNS + 2):
        msgs.append(HumanMessage(content=f"q{i}"))
        msgs.append(AIMessage(content="", tool_calls=[
            {"name": "retrieve_context", "args": {"q": f"q{i}"}, "id": f"t{i}"}
        ]))
        msgs.append(ToolMessage(content=f"ctx{i}", tool_call_id=f"t{i}"))
        msgs.append(AIMessage(content=f"a{i}"))
    out = _trim_to_recent_turns({"messages": msgs})["llm_input_messages"]
    assert isinstance(out[0], HumanMessage)
    # No ToolMessage appears before the first AIMessage carrying tool_calls.
    first_ai_toolcall = next(
        idx for idx, m in enumerate(out)
        if isinstance(m, AIMessage) and m.tool_calls
    )
    first_toolmsg = next(
        idx for idx, m in enumerate(out) if isinstance(m, ToolMessage)
    )
    assert first_ai_toolcall < first_toolmsg
