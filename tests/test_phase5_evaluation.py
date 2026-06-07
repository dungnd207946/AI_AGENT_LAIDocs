"""Phase 5 — RAG evaluation harness.

Offline tests: no LLM/network. The deterministic retrieval metrics are pure
functions; the LLM-judged metrics (grounding/NLI, answer/context relevance)
are exercised with an injected *fake judge* so the prompt-assembly and
score-aggregation logic is verified without any model call.
"""

from __future__ import annotations

import os
import tempfile

# Isolate ~/.laidocs to a temp dir BEFORE importing backend modules.
_TMP = tempfile.mkdtemp(prefix="laidocs-phase5-")
os.environ["HOME"] = _TMP
os.environ["USERPROFILE"] = _TMP

from backend.services import evaluation as ev  # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic retrieval metrics
# ---------------------------------------------------------------------------


def test_precision_recall_hit_basic():
    ranked = ["a", "b", "c", "d", "e"]
    gold = ["b", "e", "z"]  # z never retrieved
    assert ev.precision_at_k(ranked, gold, 5) == 2 / 5
    assert ev.recall_at_k(ranked, gold, 5) == 2 / 3
    assert ev.hit_at_k(ranked, gold, 5) == 1.0
    assert ev.hit_at_k(ranked, ["z"], 5) == 0.0


def test_precision_at_k_respects_cutoff():
    ranked = ["x", "a", "b"]  # only x in top-1, a/b are gold
    gold = ["a", "b"]
    assert ev.precision_at_k(ranked, gold, 1) == 0.0
    assert ev.precision_at_k(ranked, gold, 3) == 2 / 3


def test_reciprocal_rank():
    assert ev.reciprocal_rank(["a", "b", "c"], ["c"]) == 1 / 3
    assert ev.reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
    assert ev.reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_ndcg_perfect_and_imperfect():
    # Perfect ordering → nDCG 1.0
    assert ev.ndcg_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    # Gold at rank 2 only → less than a hit at rank 1
    worse = ev.ndcg_at_k(["x", "a"], ["a"], 2)
    better = ev.ndcg_at_k(["a", "x"], ["a"], 2)
    assert 0.0 < worse < better == 1.0


def test_score_retrieval_aggregate_shape():
    s = ev.score_retrieval(["a", "b", "c"], ["b"], k=3)
    d = s.as_dict()
    assert set(d) == {"precision", "recall", "hit", "mrr", "ndcg"}
    assert d["hit"] == 1.0
    assert d["mrr"] == 0.5  # gold at rank 2


def test_empty_gold_is_zero_not_crash():
    assert ev.recall_at_k(["a"], [], 5) == 0.0
    assert ev.ndcg_at_k(["a"], [], 5) == 0.0


# ---------------------------------------------------------------------------
# Grounding / faithfulness with an injected fake judge
# ---------------------------------------------------------------------------


def _make_fake_judge(claims, verdicts=None, relevance=None):
    """Build a fake judge dispatching on the prompt's task.

    ``verdicts`` maps a substring of a claim → "supported"/"unsupported".
    Unlisted claims default to "supported".
    """
    verdicts = verdicts or {}

    def judge(prompt: str) -> dict:
        if "atomic" in prompt:  # claim-decomposition prompt
            return {"claims": claims}
        if "natural-language-inference" in prompt:  # NLI prompt
            for needle, verdict in verdicts.items():
                if needle in prompt:
                    return {"verdict": verdict, "reason": "test"}
            return {"verdict": "supported", "reason": "test"}
        if "relevance" in prompt:  # relevance prompts
            return {"relevance": relevance if relevance is not None else 1.0}
        return {}

    return judge


def test_grounding_all_supported():
    judge = _make_fake_judge(["The sky is blue.", "Water is wet."])
    r = ev.verify_grounding("ignored", "ctx", judge=judge)
    assert r.total_claims == 2
    assert r.supported_claims == 2
    assert r.score == 1.0
    assert r.unsupported == []


def test_grounding_flags_unsupported_claim():
    judge = _make_fake_judge(
        ["Revenue grew 10%.", "The CEO is named Bob."],
        verdicts={"Bob": "unsupported"},
    )
    r = ev.verify_grounding("ignored", "ctx", judge=judge)
    assert r.total_claims == 2
    assert r.supported_claims == 1
    assert r.score == 0.5
    assert r.unsupported == ["The CEO is named Bob."]


def test_grounding_no_claims_scores_one():
    judge = _make_fake_judge([])  # answer with no checkable claims
    r = ev.verify_grounding("Thanks!", "ctx", judge=judge)
    assert r.total_claims == 0
    assert r.score == 1.0


def test_faithfulness_alias_matches_grounding():
    judge = _make_fake_judge(["A.", "B."], verdicts={"B.": "unsupported"})
    assert ev.faithfulness("ignored", "ctx", judge=judge) == 0.5


def test_no_judge_available_is_neutral():
    # No judge injected and LLM not configured → neutral, never false-positive.
    r = ev.verify_grounding("Some claim.", "ctx", judge=None, settings=None)
    assert r.score == 1.0 and r.total_claims == 0
    assert ev.answer_relevance("q", "a", judge=None, settings=None) == 0.0


# ---------------------------------------------------------------------------
# Relevance metrics
# ---------------------------------------------------------------------------


def test_answer_relevance_clamped():
    judge = lambda p: {"relevance": 1.7}  # out-of-range  # noqa: E731
    assert ev.answer_relevance("q", "a", judge=judge) == 1.0
    judge2 = lambda p: {"relevance": -0.5}  # noqa: E731
    assert ev.answer_relevance("q", "a", judge=judge2) == 0.0
    judge3 = lambda p: {"relevance": "garbage"}  # noqa: E731
    assert ev.answer_relevance("q", "a", judge=judge3) == 0.0


def test_context_relevance_passthrough():
    judge = lambda p: {"relevance": 0.42}  # noqa: E731
    assert ev.context_relevance("q", "ctx", judge=judge) == 0.42


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------


def test_evaluate_dataset_deterministic_only():
    # No judge, judged=False → only retrieval metrics, fully offline.
    cases = [
        ev.EvalCase(question="q1", gold_units=["a"], ranked_units=["a", "b"]),
        ev.EvalCase(question="q2", gold_units=["x"], ranked_units=["y", "x"]),
    ]
    out = ev.evaluate_dataset(cases, k=2, judged=False)
    assert len(out["cases"]) == 2
    assert "retrieval" in out["cases"][0]
    # q1 hit@1 rank1, q2 hit rank2 → mean MRR = (1 + 0.5)/2
    assert out["aggregate"]["mrr"] == 0.75
    assert out["aggregate"]["hit"] == 1.0
    # No judged metrics present.
    assert "faithfulness" not in out["aggregate"]


def test_evaluate_dataset_with_fake_judge():
    judge = _make_fake_judge(["claim one."], relevance=0.8)
    cases = [
        ev.EvalCase(
            question="q1",
            gold_units=["a"],
            ranked_units=["a"],
            context="some context",
            answer="claim one.",
        ),
    ]
    out = ev.evaluate_dataset(cases, k=1, judge=judge)
    row = out["cases"][0]
    assert row["retrieval"]["precision"] == 1.0
    assert row["faithfulness"] == 1.0
    assert row["answer_relevance"] == 0.8
    assert row["context_relevance"] == 0.8
    agg = out["aggregate"]
    assert agg["faithfulness"] == 1.0
    assert agg["answer_relevance"] == 0.8
    assert agg["precision"] == 1.0


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
