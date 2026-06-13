"""Regression test: the retrieve_context tool must record citation evidence.

Citations in the chat UI are driven by `get_retrieved_evidence()`, which reads
`ctx["retrieved_units"]`. The multi-doc retrieval refactor briefly broke this by
switching the tool to `agentic_retrieve_context_multi` (context only, no
evidence), so answers rendered with no citation chips. These tests pin the wiring
so evidence is recorded — and accumulated across calls — for the live agent.
"""

from backend.services import agent as A


def _set_ctx(monkeypatch, fake_return):
    """Point the tool at a fake retrieval and set a minimal tool context."""
    monkeypatch.setattr(
        A.retrieval,
        "agentic_retrieve_context_multi_with_evidence",
        lambda doc_ids, question, settings: fake_return,
    )
    A.set_tool_context(["docA"], settings=object(), doc_titles={"docA": "Alpha"})


def _call(question: str) -> str:
    # `retrieve_context` is a LangChain StructuredTool; call the wrapped fn.
    return A.retrieve_context.invoke({"question": question})


def test_retrieve_context_records_evidence(monkeypatch):
    evidence = [
        {"unit_id": "docA::u1", "title": "Intro", "kind": "text",
         "heading_path": ["Intro"], "preview": "hello"},
        {"unit_id": "docA::u2", "title": "Body", "kind": "text",
         "heading_path": ["Body"], "preview": "world"},
    ]
    _set_ctx(monkeypatch, ("Some context", evidence))

    out = _call("what is this")

    assert out == "Some context"
    recorded = A.get_retrieved_evidence()
    assert [u["unit_id"] for u in recorded] == ["docA::u1", "docA::u2"]


def test_evidence_accumulates_and_dedupes_across_calls(monkeypatch):
    A.set_tool_context(["docA"], settings=object(), doc_titles={})

    first = [{"unit_id": "u1", "preview": "a"}, {"unit_id": "u2", "preview": "b"}]
    monkeypatch.setattr(
        A.retrieval, "agentic_retrieve_context_multi_with_evidence",
        lambda d, q, s: ("ctx1", first),
    )
    _call("q1")

    # Second call returns one overlapping unit (u2) and one new (u3).
    second = [{"unit_id": "u2", "preview": "b"}, {"unit_id": "u3", "preview": "c"}]
    monkeypatch.setattr(
        A.retrieval, "agentic_retrieve_context_multi_with_evidence",
        lambda d, q, s: ("ctx2", second),
    )
    _call("q2")

    recorded = [u["unit_id"] for u in A.get_retrieved_evidence()]
    assert recorded == ["u1", "u2", "u3"]  # u2 not duplicated


def test_no_evidence_leaves_units_empty(monkeypatch):
    _set_ctx(monkeypatch, ("", []))
    out = _call("out of scope")
    assert out.startswith("No relevant sections")
    assert A.get_retrieved_evidence() == []
