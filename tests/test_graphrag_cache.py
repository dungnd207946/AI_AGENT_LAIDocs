"""GraphRAG live-path cache — persistence, incremental re-extraction, fusion.

Offline: no LLM/network. A deterministic *fake extractor* stands in for the
LLM so triple-extraction wiring, the SQLite triple cache, incremental rebuilds
keyed by unit_hash, and the cache-backed query path are all verified without a
model. Mirrors the isolation pattern in test_phase6_knowledge_graph.py.
"""

from __future__ import annotations

import os
import tempfile

# Isolate ~/.laidocs to a temp dir BEFORE importing backend modules.
_TMP = tempfile.mkdtemp(prefix="laidocs-graphrag-")
os.environ["HOME"] = _TMP
os.environ["USERPROFILE"] = _TMP

from backend.core.database import get_db, init_db  # noqa: E402
from backend.services import knowledge_graph as kg  # noqa: E402

init_db()

# Three units: u1 + u2 form a 2-hop chain (Acme -> Jane -> Paris); u3 is noise.
UNITS = [
    {"unit_id": "u1", "kind": "text", "title": "Company", "text": "Acme Corp was founded by Jane Doe."},
    {"unit_id": "u2", "kind": "text", "title": "Bio", "text": "Jane Doe was born in Paris."},
    {"unit_id": "u3", "kind": "text", "title": "Misc", "text": "The weather today is sunny."},
]

# Ground-truth triples per distinctive text snippet.
_FACTS = {
    "founded by Jane Doe": ("Acme Corp", "founded by", "Jane Doe"),
    "born in Paris": ("Jane Doe", "born in", "Paris"),
}


def _make_counting_extractor():
    """Fake extractor: deterministic triples/entities + an extraction call counter."""
    calls = {"triples": 0, "entities": 0}

    def _extract(prompt: str) -> dict:
        if '"entities"' in prompt:  # query-entity prompt
            calls["entities"] += 1
            ents = [e for e in ("Acme Corp", "Paris", "Jane Doe") if e in prompt]
            return {"entities": ents}
        # triple-extraction prompt
        calls["triples"] += 1
        triples = [
            {"subject": s, "relation": r, "object": o}
            for snippet, (s, r, o) in _FACTS.items()
            if snippet in prompt
        ]
        return {"triples": triples}

    return _extract, calls


def _settings():
    from backend.core.config import get_settings
    return get_settings()


# ---------------------------------------------------------------------------
# Persistence + load
# ---------------------------------------------------------------------------


def test_ensure_graph_index_persists_and_load_reconstructs():
    doc = "doc-persist"
    extractor, _ = _make_counting_extractor()
    assert kg.ensure_graph_index(doc, _settings(), units=UNITS, extractor=extractor) is True

    # Rows are persisted, one per unit (including the no-triple unit u3).
    with get_db() as conn:
        rows = conn.execute(
            "SELECT unit_id, triples FROM document_graph_units WHERE doc_id=?", (doc,)
        ).fetchall()
    assert {r["unit_id"] for r in rows} == {"u1", "u2", "u3"}

    graph = kg.load_graph(doc, _settings())
    ents = set(graph.entities())
    assert {"acme corp", "jane doe", "paris"} <= ents
    assert graph.unit_index["jane doe"] == {"u1", "u2"}


# ---------------------------------------------------------------------------
# Incremental re-extraction (the cache's whole point)
# ---------------------------------------------------------------------------


def test_unchanged_units_are_not_reextracted():
    doc = "doc-incremental-noop"
    extractor, calls = _make_counting_extractor()
    kg.ensure_graph_index(doc, _settings(), units=UNITS, extractor=extractor)
    first = calls["triples"]
    assert first == 3  # one extraction per unit on a cold cache

    # Second call, identical units → zero re-extraction.
    kg.ensure_graph_index(doc, _settings(), units=UNITS, extractor=extractor)
    assert calls["triples"] == first


def test_only_changed_unit_is_reextracted():
    doc = "doc-incremental-edit"
    extractor, calls = _make_counting_extractor()
    kg.ensure_graph_index(doc, _settings(), units=UNITS, extractor=extractor)
    baseline = calls["triples"]

    edited = [dict(u) for u in UNITS]
    edited[2] = {**edited[2], "text": "Updated: nothing relational here either."}
    kg.ensure_graph_index(doc, _settings(), units=edited, extractor=extractor)
    assert calls["triples"] == baseline + 1  # only u3 re-extracted


# ---------------------------------------------------------------------------
# Cache-backed query path (graph walk → fused unit ids)
# ---------------------------------------------------------------------------


def test_graph_augmented_units_cached_multi_hop(monkeypatch):
    extractor, _ = _make_counting_extractor()
    # Offline: bypass the LLM-backed resolver and the DB-backed unit loader.
    monkeypatch.setattr(kg, "_resolve_extractor", lambda e, s: extractor)
    monkeypatch.setattr(kg.retrieval, "get_retrieval_units", lambda doc_id: UNITS)

    ids = kg.graph_augmented_units_cached(
        "doc-query", "How is Acme Corp connected to Paris?", _settings(), hops=2
    )
    # The 2-hop walk surfaces the chain's source passages, not the noise unit.
    assert set(ids) == {"u1", "u2"}
    assert "u3" not in ids
