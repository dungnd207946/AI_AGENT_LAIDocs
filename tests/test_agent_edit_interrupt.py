"""Regression tests for the apply_edit human-in-the-loop confirmation gate.

`apply_edit` is a HARD gate: it calls LangGraph `interrupt()` to pause the graph
BEFORE writing, surfaces the diff, and persists only after the user resumes with
"approve". These tests pin that contract end-to-end by driving the REAL
`apply_edit` tool inside a minimal ToolNode graph (the same node type
create_react_agent runs tools in), with a real checkpointer:

  - the turn pauses with the exact diff payload the chat stream emits as [INTERRUPT]
  - resume "approve" writes through persist_document_content; "reject" does not
  - the chat.py guards that keep a thread from wedging on a stale interrupt

Document I/O is stubbed so the test stays focused on the gate, not the vault.
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from backend.services import agent as A
from backend.api import chat as C

DOC = "# Title\n\nHello world. Remove me please.\n"
EDIT_ARGS = {"file": "Alpha", "old_string": "Remove me please.", "new_string": ""}


def _setup(monkeypatch):
    """Stub document I/O + a tool context; return the list persist writes land in."""
    monkeypatch.setattr(A, "_get_document_content", lambda doc_id: DOC)
    persisted: list[tuple[str, str]] = []

    async def fake_persist(doc_id, content):
        persisted.append((doc_id, content))

    monkeypatch.setattr(A, "persist_document_content", fake_persist)
    A.set_tool_context(["docA"], settings=object(), doc_titles={"docA": "Alpha"})
    return persisted


def _build_graph():
    """model → tools(apply_edit) → model loop, like a 1-tool ReAct agent."""

    def model_node(state):
        if not any(m.type == "tool" for m in state["messages"]):
            return {"messages": [AIMessage(content="", tool_calls=[
                {"name": "apply_edit", "args": EDIT_ARGS, "id": "call-1"}
            ])]}
        return {"messages": [AIMessage(content="done")]}

    g = StateGraph(MessagesState)
    g.add_node("model", model_node)
    g.add_node("tools", ToolNode([A.apply_edit]))
    g.add_edge(START, "model")
    g.add_conditional_edges(
        "model",
        lambda s: "tools" if getattr(s["messages"][-1], "tool_calls", None) else END,
        {"tools": "tools", END: END},
    )
    g.add_edge("tools", "model")
    return g.compile(checkpointer=InMemorySaver())


def test_apply_edit_pauses_with_diff_and_writes_nothing(monkeypatch):
    persisted = _setup(monkeypatch)
    app = _build_graph()

    async def run():
        cfg = {"configurable": {"thread_id": "t-pause"}}
        await app.ainvoke({"messages": [HumanMessage(content="delete that line")]}, config=cfg)
        return await app.aget_state(cfg)

    snap = asyncio.run(run())

    assert snap.interrupts, "graph must pause at the edit-confirmation gate"
    payload = snap.interrupts[0].value  # exactly what chat.py emits as [INTERRUPT]
    assert payload == {
        "type": "edit_confirmation",
        "file": "Alpha",
        "action": "DELETE",
        "old_string": "Remove me please.",
        "new_string": "",
    }
    assert persisted == [], "nothing may be written while awaiting approval"
    assert A.document_was_edited() is False


def test_resume_approve_persists_the_edit(monkeypatch):
    persisted = _setup(monkeypatch)
    app = _build_graph()

    async def run():
        cfg = {"configurable": {"thread_id": "t-approve"}}
        await app.ainvoke({"messages": [HumanMessage(content="delete")]}, config=cfg)
        await app.ainvoke(Command(resume="approve"), config=cfg)
        return await app.aget_state(cfg)

    snap = asyncio.run(run())

    assert not snap.interrupts, "the gate must be cleared after resume"
    assert len(persisted) == 1, "approve must write exactly once"
    doc_id, new_content = persisted[0]
    assert doc_id == "docA"
    assert "Remove me please." not in new_content
    tool_msgs = [m.content for m in snap.values["messages"] if m.type == "tool"]
    assert any("applied successfully" in c.lower() for c in tool_msgs)
    assert A.document_was_edited() is True


def test_resume_reject_leaves_document_untouched(monkeypatch):
    persisted = _setup(monkeypatch)
    app = _build_graph()

    async def run():
        cfg = {"configurable": {"thread_id": "t-reject"}}
        await app.ainvoke({"messages": [HumanMessage(content="delete")]}, config=cfg)
        await app.ainvoke(Command(resume="reject"), config=cfg)
        return await app.aget_state(cfg)

    snap = asyncio.run(run())

    assert not snap.interrupts
    assert persisted == [], "reject must not write"
    tool_msgs = [m.content for m in snap.values["messages"] if m.type == "tool"]
    assert any("rejected" in c.lower() for c in tool_msgs)
    assert A.document_was_edited() is False


def test_clear_pending_interrupt_unwedges_thread(monkeypatch):
    """The /stream guard: a stale gate the user walked away from is discarded
    (edit never applied) so a new question isn't blocked. Exercises the real
    chat.py helpers against the same graph."""
    persisted = _setup(monkeypatch)
    app = _build_graph()

    async def run():
        cfg = {"configurable": {"thread_id": "t-stale"}}
        await app.ainvoke({"messages": [HumanMessage(content="delete")]}, config=cfg)
        # A plain new input does NOT clear a pending interrupt — it stays wedged.
        assert await C._has_pending_interrupt(app, cfg) is True
        await C._clear_pending_interrupt(app, cfg)
        return await C._has_pending_interrupt(app, cfg)

    still_pending = asyncio.run(run())

    assert still_pending is False, "the thread must be clean after clearing"
    assert persisted == [], "the abandoned edit must not have been applied"
