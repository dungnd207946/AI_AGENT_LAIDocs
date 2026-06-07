"""Phase 4 — multimodal retrieval units (figures + tables).

Offline tests: no LLM/network. They exercise the pure parsing/assembly layer
in ``backend.services.retrieval`` plus its integration into the shared unit
corpus and the labelled context builder.
"""

from __future__ import annotations

import os
import tempfile

# Isolate ~/.laidocs to a temp dir BEFORE importing backend modules.
_TMP = tempfile.mkdtemp(prefix="laidocs-phase4-")
os.environ["HOME"] = _TMP
os.environ["USERPROFILE"] = _TMP

from backend.services import retrieval  # noqa: E402


DOC_WITH_FIGURE = """\
# Results

The model converges quickly.

![Image 1](/assets/abc_1.png)
> **Description:** A line chart showing training loss decreasing from 2.5 to 0.3 over 50 epochs.

Figure 1: Training loss curve for the proposed architecture.

Some trailing prose paragraph that is unrelated to the figure entirely here.
"""

DOC_WITH_TABLE = """\
# Benchmark

Table 2: Accuracy comparison across datasets.

| Model    | CIFAR-10 | ImageNet |
| -------- | -------- | -------- |
| Baseline | 91.2     | 76.5     |
| Ours     | 94.8     | 81.3     |

The proposed model outperforms the baseline on both benchmarks.
"""

DOC_GENERIC_IMAGE = """\
# Cover

![Image 1](/assets/x_1.png)

Just an unrelated paragraph with no caption near the logo image above it.
"""


def test_image_unit_extracted_with_description_and_caption():
    units = retrieval._extract_image_units(DOC_WITH_FIGURE)
    assert len(units) == 1
    u = units[0]
    assert u["unit_id"] == "img0001"
    assert u["kind"] == "image"
    assert "training loss" in u["text"].lower()
    # Explicit "Figure 1:" caption preferred for the title.
    assert u["title"].lower().startswith("figure 1")


def test_generic_image_without_caption_is_skipped():
    # An image whose only signal is "Image 1" alt + no description/caption
    # must not pollute the corpus.
    assert retrieval._extract_image_units(DOC_GENERIC_IMAGE) == []


def test_table_unit_keeps_markdown_and_caption():
    units = retrieval._extract_table_units(DOC_WITH_TABLE)
    assert len(units) == 1
    u = units[0]
    assert u["unit_id"] == "tbl0001"
    assert u["kind"] == "table"
    assert u["title"].lower().startswith("table 2")
    # Intact cells preserved for table QA.
    assert "94.8" in u["text"] and "CIFAR-10" in u["text"]
    assert u["text"].count("|") > 6  # the pipe table survived


def test_no_false_positive_table_on_plain_pipes():
    # A line with a pipe but no separator row is not a table.
    content = "Use a | b to pipe.\n\nNo table here at all in this paragraph."
    assert retrieval._extract_table_units(content) == []


def test_build_context_labels_figures_and_tables():
    units = [
        {"unit_id": "0001", "title": "Intro", "text": "hello", "kind": "text"},
        {"unit_id": "img0001", "title": "Figure 1", "text": "a chart", "kind": "image"},
        {"unit_id": "tbl0001", "title": "Table 2", "text": "| a |\n| - |", "kind": "table"},
    ]
    ctx = retrieval.build_context_from_units(units)
    assert "[Section: Intro (node 0001)]" in ctx
    assert "[Figure: Figure 1 (img0001)]" in ctx
    assert "[Table: Table 2 (tbl0001)]" in ctx


def test_get_retrieval_units_merges_multimodal(monkeypatch):
    # No tree, content has a figure + a table → chunks + img + tbl units.
    doc = DOC_WITH_FIGURE + "\n" + DOC_WITH_TABLE
    monkeypatch.setattr(retrieval, "get_tree_index", lambda d: None)
    monkeypatch.setattr(retrieval, "get_document_content", lambda d: doc)

    units = retrieval.get_retrieval_units("doc1")
    ids = {u["unit_id"] for u in units}
    kinds = {u["kind"] for u in units}
    assert any(i.startswith("c") for i in ids)      # text chunks
    assert "img0001" in ids                          # figure unit
    assert "tbl0001" in ids                          # table unit
    assert kinds == {"text", "image", "table"}


def test_figure_unit_wins_when_sections_are_separated(monkeypatch):
    # The chart's distinctive terms live only in the figure description →
    # the focused figure unit must out-rank the unrelated prose sections.
    tree = {
        "structure": [
            {"node_id": "0001", "title": "Introduction",
             "text": "This paper studies neural network architectures broadly.", "nodes": []},
            {"node_id": "0002", "title": "Related Work",
             "text": "Prior approaches used recurrent and convolutional designs.", "nodes": []},
        ]
    }
    monkeypatch.setattr(retrieval, "get_tree_index", lambda d: tree)
    monkeypatch.setattr(retrieval, "get_document_content", lambda d: DOC_WITH_FIGURE)

    units = retrieval.get_retrieval_units("doc1")
    ranked = retrieval.bm25_search("doc1", "training loss curve over epochs", units=units)
    assert ranked and ranked[0] == "img0001", f"expected figure top-ranked, got {ranked}"


def test_table_unit_wins_when_sections_are_separated(monkeypatch):
    # Realistic case: a tree-structured doc whose sections are distinct units.
    # The distinctive term lives only in the table → the focused table unit
    # must out-rank the unrelated prose sections.
    tree = {
        "structure": [
            {"node_id": "0001", "title": "Introduction",
             "text": "This paper studies neural network training dynamics.", "nodes": []},
            {"node_id": "0002", "title": "Method",
             "text": "We propose a new optimizer with adaptive momentum.", "nodes": []},
        ]
    }
    monkeypatch.setattr(retrieval, "get_tree_index", lambda d: tree)
    monkeypatch.setattr(retrieval, "get_document_content", lambda d: DOC_WITH_TABLE)

    units = retrieval.get_retrieval_units("doc1")
    ranked = retrieval.bm25_search("doc1", "accuracy on CIFAR-10 and ImageNet", units=units)
    assert ranked and ranked[0] == "tbl0001", f"expected table top-ranked, got {ranked}"


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
