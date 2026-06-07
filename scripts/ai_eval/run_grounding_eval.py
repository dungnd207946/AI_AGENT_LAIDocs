#!/usr/bin/env python3
"""Answer-quality evaluation — grounding / faithfulness + relevance.

Runs the RAGAS-style LLM-judged metrics from ``backend.services.evaluation``
over a dataset of (question, context, answer) triples:

  * faithfulness   — fraction of the answer's atomic claims entailed by the
                     retrieved context (the hallucination signal)
  * answer_relevance  — does the answer address the question?
  * context_relevance — did retrieval surface relevant context?

Every LLM call goes through the injectable ``judge`` callable. By default a
provider-agnostic judge is built from your configured Settings (needs an LLM).
Pass ``--dry-run`` to verify the dataset wiring with a neutral no-LLM judge
(faithfulness reports 1.0, relevance 0.0 — see evaluation.py docstrings).

Dataset format (JSON list)
--------------------------
    [
      {
        "question": "What is the refund window?",
        "context":  "[Section: Refunds] Refunds are available within 30 days...",
        "answer":   "You can request a refund within 30 days of purchase.",
        "gold_units":   ["0007"],            # optional → retrieval metrics too
        "ranked_units": ["0007", "0003"]     # optional
      }
    ]

Usage
-----
    .venv-ai/Scripts/python.exe scripts/ai_eval/run_grounding_eval.py \
        --dataset scripts/ai_eval/datasets/sample_eval.json

    # offline wiring check, no model calls:
    ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services import evaluation as ev  # noqa: E402


def _neutral_judge(_prompt: str) -> dict:
    """A no-op judge for --dry-run: returns nothing so scores stay neutral."""
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--ranked-key", default="hybrid",
                    help="which variant to score when ranked_units is a "
                         "per-variant dict (default: hybrid)")
    ap.add_argument("--dry-run", action="store_true",
                    help="use a neutral no-LLM judge (wiring check only)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    raw = json.loads(args.dataset.read_text(encoding="utf-8"))

    def _ranked(c: dict) -> list:
        """A case may carry a flat list or a {variant: [...]} dict."""
        ru = c.get("ranked_units", [])
        if isinstance(ru, dict):
            return ru.get(args.ranked_key) or next(iter(ru.values()), [])
        return ru

    cases = [
        ev.EvalCase(
            question=c.get("question", ""),
            gold_units=c.get("gold_units", []),
            ranked_units=_ranked(c),
            context=c.get("context", ""),
            answer=c.get("answer", ""),
        )
        for c in raw
    ]

    judge = _neutral_judge if args.dry_run else None
    result = ev.evaluate_dataset(cases, k=args.k, judge=judge, judged=True)

    print(f"\nAnswer-quality evaluation (k={result['k']}, n={len(cases)}, "
          f"{'DRY-RUN neutral judge' if args.dry_run else 'LLM judge'})")
    print("Aggregate:")
    for metric, value in sorted(result["aggregate"].items()):
        print(f"  {metric:<18} {value:.4f}")

    # Surface flagged hallucinations per case.
    flagged = [(r["question"], r["unsupported_claims"])
               for r in result["cases"] if r.get("unsupported_claims")]
    if flagged:
        print("\nUnsupported claims (potential hallucinations):")
        for q, claims in flagged:
            print(f"  Q: {q}")
            for c in claims:
                print(f"     - {c}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote report -> {args.out}")


if __name__ == "__main__":
    main()
