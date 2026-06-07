"""Phase 6 — knowledge graph + graph-of-thought.

Offline tests: no LLM/network. The deterministic graph core (build, k-hop
subgraph, path finding, unit mapping, GoT rendering) is tested directly; the
LLM-dependent extraction is exercised through an injected *fake extractor* so
prompt dispatch and end-to-end wiring are verified without a model.
"""

from __future__ import annotations

import os
import tempfile

# Isolate ~/.laidocs to a temp dir BEFORE importing backend modules.
_TMP = tempfile.mkdtemp(prefix="laidocs-phase6-")
os.environ["HOME"] = _TMP
os.environ["USERPROFILE"] = _TMP

from backend.services import knowledge_graph as kg  # noqa: E402


# A small fact set: A founded B; B located in C; D works at B.
TRIPLES = [
    kg.Triple("Acme Corp", "founded by", "Jane Doe", "u1"),
    kg.Triple("Acme Corp", "located in", "Berlin", "u1"),
    kg.Triple("John Smith", "works at", "Acme Corp", "u2"),
    kg.Triple("Jane Doe", "born in", "Paris", "u3"),
]


def _graph():
    return kg.build_graph(TRIPLES)


# ---------------------------------------------------------------------------
# Construction & normalization
# ---------------------------------------------------------------------------


def test_build_graph_indexes_entities_and_units():
    g = _graph()
    ents = set(g.entities())
    assert "acme corp" in ents and "jane doe" in ents and "berlin" in ents
    # unit_index bridges entities back to source passages.
    assert g.unit_index["acme corp"] == {"u1", "u2"}
    assert g.unit_index["jane doe"] == {"u1", "u3"}
    # display labels preserved.
    assert g.label("acme corp") == "Acme Corp"


def test_neighbors_directed_edges():
    g = _graph()
    rels = {(rel, tgt) for rel, tgt, _ in g.neighbors("Acme Corp")}
    assert ("founded by", "jane doe") in rels
    assert ("located in", "berlin") in rels


def test_empty_subject_object_skipped():
    g = kg.build_graph([kg.Triple("", "rel", "x", "u"), kg.Triple("a", "rel", "", "u")])
    assert g.entities() == []


# ---------------------------------------------------------------------------
# Entity matching
# ---------------------------------------------------------------------------


def test_match_entities_exact_and_substring():
    g = _graph()
    # exact (case-insensitive)
    assert kg.match_entities(g, ["ACME CORP"]) == ["acme corp"]
    # substring containment either direction
    matched = kg.match_entities(g, ["Acme"])
    assert "acme corp" in matched


def test_match_entities_dedup_order():
    g = _graph()
    out = kg.match_entities(g, ["Acme Corp", "acme corp", "Jane Doe"])
    assert out == ["acme corp", "jane doe"]


# ---------------------------------------------------------------------------
# k-hop subgraph
# ---------------------------------------------------------------------------


def test_one_hop_from_acme():
    g = _graph()
    reached, edges = kg.k_hop_subgraph(g, ["Acme Corp"], hops=1)
    # 1 hop reaches founder, city, and the employee (incoming edge).
    assert {"acme corp", "jane doe", "berlin", "john smith"} <= reached
    # Paris is 2 hops away (via Jane Doe) → not reached at hop 1.
    assert "paris" not in reached


def test_two_hops_reaches_paris():
    g = _graph()
    reached, _ = kg.k_hop_subgraph(g, ["Acme Corp"], hops=2)
    assert "paris" in reached


def test_zero_hops_is_just_seed():
    g = _graph()
    reached, edges = kg.k_hop_subgraph(g, ["Acme Corp"], hops=0)
    assert reached == {"acme corp"}
    assert edges == []


# ---------------------------------------------------------------------------
# Path finding & graph-of-thought
# ---------------------------------------------------------------------------


def test_find_paths_connects_employee_to_city():
    g = _graph()
    # John Smith -> Acme Corp -> Berlin
    paths = kg.find_paths(g, "John Smith", "Berlin")
    assert paths
    # The shortest path is two hops through Acme Corp.
    shortest = min(paths, key=len)
    assert len(shortest) == 2
    mids = {hop[2] for hop in shortest[:-1]}  # intermediate targets
    assert "Acme Corp" in mids


def test_find_paths_no_connection_returns_empty():
    g = kg.build_graph([kg.Triple("X", "rel", "Y", "u"), kg.Triple("P", "rel", "Q", "u")])
    assert kg.find_paths(g, "X", "Q") == []


def test_reason_paths_and_render():
    g = _graph()
    paths = kg.reason_paths(g, ["John Smith", "Berlin"])
    assert paths
    text = kg.render_reasoning(paths)
    assert "Reasoning paths" in text
    assert "John Smith" in text and "Berlin" in text
    assert "--[" in text  # relation markers rendered


def test_render_empty_paths():
    assert kg.render_reasoning([]) == ""


# ---------------------------------------------------------------------------
# Unit mapping
# ---------------------------------------------------------------------------


def test_units_for_entities():
    g = _graph()
    uids = kg.units_for_entities(g, ["Acme Corp"])
    assert uids == ["u1", "u2"]
    uids2 = kg.units_for_entities(g, ["Jane Doe", "John Smith"])
    assert set(uids2) == {"u1", "u2", "u3"}


# ---------------------------------------------------------------------------
# Triple/query extraction via fake extractor
# ---------------------------------------------------------------------------


def _fake_extractor(triples_by_text=None, query_entities=None):
    """Dispatch on prompt type. ``triples_by_text`` maps a substring of the
    unit text → list of triple dicts."""
    triples_by_text = triples_by_text or {}

    def extract(prompt: str) -> dict:
        if "(subject, relation, object)" in prompt:
            for needle, triples in triples_by_text.items():
                if needle in prompt:
                    return {"triples": triples}
            return {"triples": []}
        if "named entities" in prompt:
            return {"entities": query_entities or []}
        return {}

    return extract


def test_extract_triples_from_units_tags_unit_id():
    units = [
        {"unit_id": "0001", "title": "Founders", "text": "Acme was founded by Jane."},
    ]
    extractor = _fake_extractor(
        triples_by_text={"Acme was founded": [
            {"subject": "Acme", "relation": "founded by", "object": "Jane"},
        ]}
    )
    triples = kg.extract_triples_from_units(units, extractor)
    assert len(triples) == 1
    assert triples[0].subject == "Acme" and triples[0].unit_id == "0001"


def test_extract_triples_skips_malformed():
    units = [{"unit_id": "u", "title": "", "text": "stuff here"}]
    extractor = _fake_extractor(triples_by_text={"stuff here": [
        {"subject": "A", "relation": "", "object": "B"},   # empty relation
        {"subject": "A", "relation": "r", "object": "B"},  # good
        "not a dict",
    ]})
    triples = kg.extract_triples_from_units(units, extractor)
    assert len(triples) == 1 and triples[0].relation == "r"


def test_graph_augmented_units_end_to_end(monkeypatch):
    units = [
        {"unit_id": "u1", "title": "", "text": "Acme Corp founded by Jane Doe in Berlin."},
        {"unit_id": "u2", "title": "", "text": "John Smith works at Acme Corp."},
        {"unit_id": "u3", "title": "", "text": "Unrelated text about cooking."},
    ]
    monkeypatch.setattr(kg.retrieval, "get_retrieval_units", lambda d: units)

    extractor = _fake_extractor(
        triples_by_text={
            "Acme Corp founded": [
                {"subject": "Acme Corp", "relation": "founded by", "object": "Jane Doe"},
            ],
            "John Smith works": [
                {"subject": "John Smith", "relation": "works at", "object": "Acme Corp"},
            ],
        },
        query_entities=["Jane Doe"],
    )

    # Question about Jane Doe → graph walk reaches Acme Corp (1 hop) and
    # John Smith (2 hops), surfacing u1 and u2; the cooking unit is excluded.
    out = kg.graph_augmented_units("doc1", "Who is Jane Doe connected to?",
                                   extractor=extractor, units=units, hops=2)
    assert "u1" in out and "u2" in out
    assert "u3" not in out


def test_graph_augmented_units_no_extractor_is_noop():
    # No extractor + no LLM configured → empty, never harms the pipeline.
    out = kg.graph_augmented_units("doc1", "q", extractor=None, settings=None,
                                   units=[{"unit_id": "u", "title": "", "text": "x"}])
    assert out == []


def test_graph_of_thought_end_to_end():
    units = [
        {"unit_id": "u1", "title": "", "text": "Acme Corp located in Berlin."},
        {"unit_id": "u2", "title": "", "text": "John Smith works at Acme Corp."},
    ]
    extractor = _fake_extractor(
        triples_by_text={
            "Acme Corp located": [
                {"subject": "Acme Corp", "relation": "located in", "object": "Berlin"},
            ],
            "John Smith works": [
                {"subject": "John Smith", "relation": "works at", "object": "Acme Corp"},
            ],
        },
        query_entities=["John Smith", "Berlin"],
    )
    got = kg.graph_of_thought("doc1", "How is John Smith related to Berlin?",
                              extractor=extractor, units=units)
    assert "Reasoning paths" in got
    assert "John Smith" in got and "Berlin" in got


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
