#!/usr/bin/env python3
"""Retrieval benchmark — compare retriever variants on a fixed dataset.

This script is the reproducible entry point for the retrieval-quality numbers in
``docs/ai/EVALUATION.md``. It scores one or more *retriever variants* against a
gold-labelled dataset using the deterministic metrics in
``backend.services.evaluation`` (precision@k / recall@k / hit@k / MRR / nDCG).

Two modes
---------
OFFLINE (default, no LLM/DB):
    Each dataset case already carries ``ranked_units`` (the ordered unit ids a
    retriever returned). The script just scores them. Fully reproducible, runs
    anywhere, used in CI.

LIVE (``--live``, needs a configured LLM + ingested docs in ~/.laidocs):
    Each case carries a ``doc_id`` and ``question``; the script *runs* the real
    retrievers (``bm25`` / ``dense`` / ``tree`` / ``hybrid`` / ``graph``) to
    produce the ranked lists, then scores each variant side by side and also
    reports per-query latency.

Dataset format (JSON list)
--------------------------
    [
      {
        "question": "What is the refund window?",
        "gold_units": ["0007", "tbl0002"],   # ids that SHOULD be retrieved
        "doc_id": "abc123",                    # required for --live
        "ranked_units": {                      # required for OFFLINE
          "bm25":   ["0007", "0003", ...],
          "dense":  ["0007", "tbl0002", ...],
          "hybrid": ["0007", "tbl0002", ...]
        }
      },
      ...
    ]

In OFFLINE mode every key under ``ranked_units`` becomes a compared variant.
A flat ``"ranked_units": [...]`` list is also accepted and scored as "default".

Usage
-----
    # offline, reproducible
    .venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
        --dataset scripts/ai_eval/datasets/sample_eval.json -k 5

    # live, runs the real retrievers against your vault
    .venv-ai/Scripts/python.exe scripts/ai_eval/run_retrieval_benchmark.py \
        --dataset my_live_set.json --live --variants bm25,dense,tree,hybrid -k 5

    # write a machine-readable report
    ... --out runs/2026-06-07_hybrid_vs_bm25.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make `backend` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services import evaluation as ev  # noqa: E402

LIVE_VARIANTS = ("bm25", "dense", "tree", "hybrid", "graph")


def _load_dataset(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Dataset must be a JSON list, got {type(data).__name__}")
    return data


def _aggregate(scores: list[ev.RetrievalScore]) -> dict[str, float]:
    if not scores:
        return {m: 0.0 for m in ("precision", "recall", "hit", "mrr", "ndcg")}
    keys = ("precision", "recall", "hit", "mrr", "ndcg")
    return {k: sum(getattr(s, k) for s in scores) / len(scores) for k in keys}


# ---------------------------------------------------------------------------
# Offline scoring (no LLM, no DB)
# ---------------------------------------------------------------------------


def run_offline(cases: list[dict], k: int) -> dict[str, dict]:
    """Score every variant found under each case's ``ranked_units``."""
    per_variant: dict[str, list[ev.RetrievalScore]] = {}
    for case in cases:
        gold = case.get("gold_units", [])
        ranked = case.get("ranked_units", {})
        if isinstance(ranked, list):
            ranked = {"default": ranked}
        for variant, ids in ranked.items():
            per_variant.setdefault(variant, []).append(
                ev.score_retrieval(ids, gold, k=k)
            )
    return {v: _aggregate(s) for v, s in per_variant.items()}


# ---------------------------------------------------------------------------
# Live scoring (runs the real retrievers)
# ---------------------------------------------------------------------------


def _live_ranked(variant: str, doc_id: str, question: str, settings, units):
    """Return ranked unit_ids for one retriever variant (+ wall-clock seconds)."""
    from backend.services import retrieval as r
    from backend.services import knowledge_graph as kg

    t0 = time.perf_counter()
    if variant == "bm25":
        out = r.bm25_search(doc_id, question, units=units)
    elif variant == "dense":
        out = r.dense_search(doc_id, question, settings, units=units)
    elif variant == "tree":
        tree = r.get_tree_index(doc_id)
        out = r.select_node_ids(tree, question, settings) if tree else []
    elif variant == "hybrid":
        out, _ = r.hybrid_rank(doc_id, question, settings, units=units)
    elif variant == "graph":
        # Cache-backed path — the same one the live agent's hybrid fusion uses
        # (persisted triples; builds incrementally on first call, fast after).
        out = kg.graph_augmented_units_cached(doc_id, question, settings)
    else:
        raise SystemExit(f"Unknown variant: {variant}")
    return out, time.perf_counter() - t0


def run_live(cases: list[dict], k: int, variants: list[str]) -> dict[str, dict]:
    from backend.core.config import get_settings
    from backend.services import retrieval as r

    settings = get_settings()
    per_variant: dict[str, list[ev.RetrievalScore]] = {}
    latencies: dict[str, list[float]] = {}

    for case in cases:
        doc_id = case.get("doc_id")
        question = case["question"]
        gold = case.get("gold_units", [])
        if not doc_id:
            raise SystemExit("--live needs a 'doc_id' on every case")
        units = r.get_retrieval_units(doc_id)
        for variant in variants:
            ranked, secs = _live_ranked(variant, doc_id, question, settings, units)
            per_variant.setdefault(variant, []).append(
                ev.score_retrieval(ranked, gold, k=k)
            )
            latencies.setdefault(variant, []).append(secs)

    report = {}
    for v, s in per_variant.items():
        agg = _aggregate(s)
        lat = latencies[v]
        agg["latency_s_mean"] = sum(lat) / len(lat) if lat else 0.0
        report[v] = agg
    return report


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_table(report: dict[str, dict], k: int) -> None:
    cols = ["precision", "recall", "hit", "mrr", "ndcg"]
    if any("latency_s_mean" in v for v in report.values()):
        cols.append("latency_s_mean")
    width = max((len(v) for v in report), default=8)
    header = f"{'variant':<{width}}  " + "  ".join(f"{c[:9]:>9}" for c in cols)
    print(f"\nRetrieval benchmark (k={k}, n_variants={len(report)})")
    print(header)
    print("-" * len(header))
    # Sort by nDCG desc for a quick leaderboard.
    for variant in sorted(report, key=lambda x: report[x].get("ndcg", 0), reverse=True):
        row = report[variant]
        cells = "  ".join(f"{row.get(c, 0.0):>9.4f}" for c in cols)
        print(f"{variant:<{width}}  {cells}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("-k", type=int, default=5, help="cutoff for @k metrics")
    ap.add_argument("--live", action="store_true",
                    help="run the real retrievers (needs LLM + ingested docs)")
    ap.add_argument("--variants", default=",".join(LIVE_VARIANTS),
                    help="comma list of live variants: " + ",".join(LIVE_VARIANTS))
    ap.add_argument("--out", type=Path, help="write JSON report here")
    args = ap.parse_args()

    cases = _load_dataset(args.dataset)
    if args.live:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        report = run_live(cases, args.k, variants)
    else:
        report = run_offline(cases, args.k)

    _print_table(report, args.k)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"k": args.k, "n_cases": len(cases), "live": args.live, "report": report},
            indent=2), encoding="utf-8")
        print(f"Wrote report -> {args.out}")


if __name__ == "__main__":
    main()
