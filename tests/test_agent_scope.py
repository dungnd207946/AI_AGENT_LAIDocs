from backend.services import agent as A


def test_resolve_scope_doc_by_title_and_id():
    ctx = {"doc_ids": ["docA", "docB"],
           "doc_titles": {"docA": "Alpha", "docB": "Beta"}}
    assert A._resolve_scope_doc("Beta", ctx) == "docB"     # by title
    assert A._resolve_scope_doc("beta", ctx) == "docB"     # case-insensitive
    assert A._resolve_scope_doc("docA", ctx) == "docA"     # by raw id


def test_resolve_scope_doc_single_scope_defaults():
    ctx = {"doc_ids": ["docA"], "doc_titles": {"docA": "Alpha"}}
    # empty file + single doc in scope → that doc
    assert A._resolve_scope_doc("", ctx) == "docA"


def test_resolve_scope_doc_rejects_out_of_scope():
    ctx = {"doc_ids": ["docA"], "doc_titles": {"docA": "Alpha"}}
    out = A._resolve_scope_doc("Gamma", ctx)
    assert out.startswith("Error")
    assert "Alpha" in out  # lists available files


def test_resolve_scope_doc_ambiguous_empty_multi():
    ctx = {"doc_ids": ["docA", "docB"],
           "doc_titles": {"docA": "Alpha", "docB": "Beta"}}
    out = A._resolve_scope_doc("", ctx)
    assert out.startswith("Error")  # must specify which file
