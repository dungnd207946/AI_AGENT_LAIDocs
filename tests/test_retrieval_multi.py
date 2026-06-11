import sqlite3
import pytest

from backend.core import database
from backend.services import retrieval as R


@pytest.fixture
def db(tmp_path, monkeypatch):
    p = tmp_path / "laidocs.db"
    monkeypatch.setattr(database, "DB_PATH", p)
    database.init_db()
    # two docs with headings so tree/units exist
    with sqlite3.connect(str(p)) as c:
        c.execute("INSERT INTO folders (path, name) VALUES ('f', 'f')")
        c.execute(
            "INSERT INTO documents (id, folder, filename, title, source_type, content) "
            "VALUES ('docA','f','a.md','Alpha','file', ?)",
            ("# Alpha\n\nApples are red and sweet.\n",),
        )
        c.execute(
            "INSERT INTO documents (id, folder, filename, title, source_type, content) "
            "VALUES ('docB','f','b.md','Beta','file', ?)",
            ("# Beta\n\nBananas are yellow.\n",),
        )
        # A third doc keeps the BM25 corpus large enough that a term appearing
        # in a single unit still earns positive IDF (N=2 gives IDF<=0).
        c.execute(
            "INSERT INTO documents (id, folder, filename, title, source_type, content) "
            "VALUES ('docC','f','c.md','Gamma','file', ?)",
            ("# Gamma\n\nCars have four wheels.\n",),
        )
        c.commit()
    return p


# ── Task 4: file label in context ───────────────────────────────────


def test_build_context_adds_file_label_when_present():
    units = [
        {"unit_id": "docA::0001", "title": "Intro", "text": "hello",
         "kind": "text", "doc_title": "Report.pdf"},
    ]
    ctx = R.build_context_from_units(units)
    assert "[File: Report.pdf" in ctx
    assert "Intro" in ctx
    # namespaced id is displayed stripped of the doc prefix
    assert "docA::" not in ctx
    assert "0001" in ctx


def test_build_context_unchanged_without_doc_title():
    units = [{"unit_id": "0001", "title": "Intro", "text": "hello", "kind": "text"}]
    ctx = R.build_context_from_units(units)
    assert ctx.startswith("[Section: Intro (node 0001)]")
    assert "File:" not in ctx


# ── Task 5: multi-doc pooled units + bm25 ────────────────────────────


def test_pool_namespaces_unit_ids_and_tags_source(db):
    units = R.get_retrieval_units_multi(["docA", "docB"])
    assert units, "expected pooled units from both docs"
    assert all("::" in u["unit_id"] for u in units)
    assert {u["doc_id"] for u in units} == {"docA", "docB"}
    assert {u["doc_title"] for u in units} == {"Alpha", "Beta"}
    for u in units:
        assert u["unit_id"].startswith(u["doc_id"] + "::")


def test_bm25_over_pool_ranks_correct_file(db):
    units = R.get_retrieval_units_multi(["docA", "docB", "docC"])
    ranked = R.bm25_search("docA", "bananas yellow", units=units)
    assert ranked, "expected a lexical hit"
    top = ranked[0]
    # the winning unit must come from docB (where 'bananas' lives)
    assert top.startswith("docB::")


# ── Task 6: agentic multi-doc retrieval ──────────────────────────────


def test_agentic_multi_no_llm_falls_back_to_pooled_singleshot(db, monkeypatch):
    import backend.services.retrieval as RR
    monkeypatch.setattr(RR, "is_llm_configured", lambda cfg: False)
    # Keep the test hermetic: skip the embedding retriever (no network).
    monkeypatch.setattr(RR, "embeddings_supported", lambda cfg: False)
    out = RR.agentic_retrieve_context_multi(["docA", "docB", "docC"], "yellow bananas")
    assert "Beta" in out  # file label of docB
    assert "Bananas are yellow" in out


def test_agentic_multi_empty_scope_returns_blank(db):
    assert R.agentic_retrieve_context_multi([], "anything") == ""
