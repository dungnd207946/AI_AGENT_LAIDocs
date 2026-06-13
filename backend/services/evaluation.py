"""RAG evaluation harness — retrieval metrics, RAGAS-style answer metrics,
and NLI-style grounding verification.

This is an AI-layer evaluation module: it scores the *quality* of the
retrieval + answer pipeline so changes (a new fusion weight, a different
embedding model, an extra agentic round) can be compared on a fixed dataset
instead of by eyeballing chat output.

Two families of metric live here:

1. **Deterministic retrieval metrics** (no LLM, fully offline): given a query
   with gold-relevant ``unit_id``s and the ranked ids a retriever returned,
   compute precision@k, recall@k, hit@k, MRR, and (optional) nDCG. These are
   the cheap, reproducible signals you run on every change.

2. **LLM-judged metrics** (RAGAS-style): faithfulness / groundedness via claim
   decomposition + NLI entailment, answer relevance, and context relevance.
   Every LLM call goes through an injectable ``judge`` callable so the whole
   module is unit-testable offline — pass a fake judge in tests; in production
   omit it and a provider-agnostic judge backed by ``llm.create_chat_model``
   is built from ``Settings``.

Nothing here touches the frontend or the FastAPI request path; it is a library
plus a callable ``evaluate_dataset`` entry point usable from a script or test.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..core.config import Settings, get_settings
from .llm import create_chat_model, is_llm_configured

logger = logging.getLogger(__name__)

# A judge takes a single prompt string and returns the parsed JSON object the
# prompt asked for. Injectable so tests can supply a deterministic stub.
Judge = Callable[[str], dict]


# ---------------------------------------------------------------------------
# Deterministic retrieval metrics
# ---------------------------------------------------------------------------


def precision_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of the top-k retrieved ids that are gold-relevant."""
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    goldset = set(gold)
    hits = sum(1 for uid in top if uid in goldset)
    return hits / len(top)


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of gold-relevant ids found within the top-k retrieved."""
    goldset = set(gold)
    if not goldset:
        return 0.0
    top = set(ranked[:k])
    return len(top & goldset) / len(goldset)


def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """1.0 if any gold id appears in the top-k, else 0.0."""
    goldset = set(gold)
    return 1.0 if goldset & set(ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[str]) -> float:
    """Reciprocal of the 1-based rank of the first gold id (0 if none)."""
    goldset = set(gold)
    for i, uid in enumerate(ranked):
        if uid in goldset:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Binary-relevance nDCG@k (all gold ids weighted equally)."""
    goldset = set(gold)
    if not goldset:
        return 0.0
    dcg = 0.0
    for i, uid in enumerate(ranked[:k]):
        if uid in goldset:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(goldset), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


@dataclass
class RetrievalScore:
    """Per-query deterministic retrieval scores."""

    precision: float
    recall: float
    hit: float
    mrr: float
    ndcg: float

    def as_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "hit": self.hit,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
        }


def score_retrieval(ranked: Sequence[str], gold: Sequence[str], k: int = 5) -> RetrievalScore:
    """Compute all deterministic retrieval metrics for one query."""
    return RetrievalScore(
        precision=precision_at_k(ranked, gold, k),
        recall=recall_at_k(ranked, gold, k),
        hit=hit_at_k(ranked, gold, k),
        mrr=reciprocal_rank(ranked, gold),
        ndcg=ndcg_at_k(ranked, gold, k),
    )


# ---------------------------------------------------------------------------
# LLM judge — provider-agnostic, injectable
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> dict:
    """Pull the first JSON object out of a model response (tolerant of prose)."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group())
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def make_llm_judge(settings: Settings, *, temperature: float = 0.0,
                   max_tokens: int = 700) -> Judge:
    """Build a JSON-returning judge backed by the configured chat model.

    The returned callable sends ``prompt`` to the model and parses the first
    JSON object from the reply. On any error it returns ``{}`` so callers can
    fall back to a neutral score rather than crash an evaluation run.
    """
    model = create_chat_model(settings.active_llm, temperature=temperature,
                              max_tokens=max_tokens)

    def _judge(prompt: str) -> dict:
        try:
            resp = model.invoke([{"role": "user", "content": prompt}])
            raw = resp.content if isinstance(resp.content, str) else str(resp.content)
            return _extract_json(raw)
        except Exception:
            logger.exception("LLM judge call failed")
            return {}

    return _judge


def _resolve_judge(judge: Judge | None, settings: Settings | None) -> Judge | None:
    """Return the judge to use, building a default from settings if possible."""
    if judge is not None:
        return judge
    settings = settings or get_settings()
    if not is_llm_configured(settings.active_llm):
        return None
    return make_llm_judge(settings)


# ---------------------------------------------------------------------------
# Grounding / faithfulness — claim decomposition + NLI entailment
# ---------------------------------------------------------------------------

_CLAIM_PROMPT = """\
Break the following ANSWER into a list of atomic, self-contained factual \
claims. Each claim must stand on its own (resolve pronouns). Ignore questions, \
hedges, and pure pleasantries.

ANSWER:
{answer}

Respond with ONLY a JSON object:
{{"claims": ["<claim 1>", "<claim 2>", ...]}}"""

_NLI_PROMPT = """\
You are a strict natural-language-inference judge. Decide whether the CONTEXT \
entails (supports) the CLAIM. Answer "supported" ONLY if the claim can be \
verified from the context alone. If the context is silent or contradicts the \
claim, answer "unsupported".

CONTEXT:
{context}

CLAIM:
{claim}

Respond with ONLY a JSON object:
{{"verdict": "supported"|"unsupported", "reason": "<short reason>"}}"""


@dataclass
class GroundingResult:
    """Outcome of grounding verification for one answer."""

    score: float  # supported_claims / total_claims (1.0 if no claims)
    total_claims: int
    supported_claims: int
    unsupported: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported": self.unsupported,
        }


def decompose_claims(answer: str, judge: Judge) -> list[str]:
    """Decompose an answer into atomic factual claims via the judge."""
    if not answer or not answer.strip():
        return []
    data = judge(_CLAIM_PROMPT.format(answer=answer.strip()))
    claims = data.get("claims") or []
    return [str(c).strip() for c in claims if str(c).strip()]


def verify_grounding(answer: str, context: str, *, judge: Judge | None = None,
                     settings: Settings | None = None) -> GroundingResult:
    """NLI-style grounding check: are the answer's claims entailed by context?

    Decomposes the answer into atomic claims, then asks the judge whether each
    is supported by the retrieved context. ``score`` is the fraction supported
    — the RAGAS "faithfulness" measure. An answer with no checkable claims
    scores 1.0 (nothing to contradict).

    Returns a neutral all-supported result if no judge is available, so an
    unconfigured environment never reports false hallucinations.
    """
    judge = _resolve_judge(judge, settings)
    if judge is None:
        return GroundingResult(score=1.0, total_claims=0, supported_claims=0)

    claims = decompose_claims(answer, judge)
    if not claims:
        return GroundingResult(score=1.0, total_claims=0, supported_claims=0)

    supported = 0
    unsupported: list[str] = []
    for claim in claims:
        data = judge(_NLI_PROMPT.format(context=context or "(empty)", claim=claim))
        verdict = str(data.get("verdict", "")).strip().lower()
        if verdict == "supported":
            supported += 1
        else:
            unsupported.append(claim)

    return GroundingResult(
        score=supported / len(claims),
        total_claims=len(claims),
        supported_claims=supported,
        unsupported=unsupported,
    )


# faithfulness is the grounding score; alias kept for RAGAS familiarity.
def faithfulness(answer: str, context: str, *, judge: Judge | None = None,
                 settings: Settings | None = None) -> float:
    """RAGAS faithfulness = fraction of answer claims grounded in context."""
    return verify_grounding(answer, context, judge=judge, settings=settings).score


# ---------------------------------------------------------------------------
# Answer relevance & context relevance (RAGAS-style)
# ---------------------------------------------------------------------------

_ANSWER_RELEVANCE_PROMPT = """\
Rate how well the ANSWER addresses the QUESTION, ignoring whether it is \
factually correct. A focused, on-topic answer scores high; an evasive, \
off-topic, or padded answer scores low.

QUESTION:
{question}

ANSWER:
{answer}

Respond with ONLY a JSON object:
{{"relevance": <float 0.0-1.0>, "reason": "<short reason>"}}"""

_CONTEXT_RELEVANCE_PROMPT = """\
Rate how relevant the retrieved CONTEXT is to answering the QUESTION. High if \
the context contains the information needed; low if it is mostly unrelated.

QUESTION:
{question}

CONTEXT:
{context}

Respond with ONLY a JSON object:
{{"relevance": <float 0.0-1.0>, "reason": "<short reason>"}}"""


def _clamp01(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def answer_relevance(question: str, answer: str, *, judge: Judge | None = None,
                     settings: Settings | None = None) -> float:
    """How well the answer addresses the question (0-1). 0.0 if no judge."""
    judge = _resolve_judge(judge, settings)
    if judge is None:
        return 0.0
    data = judge(_ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer))
    return _clamp01(data.get("relevance"))


def context_relevance(question: str, context: str, *, judge: Judge | None = None,
                      settings: Settings | None = None) -> float:
    """How relevant the retrieved context is to the question (0-1)."""
    judge = _resolve_judge(judge, settings)
    if judge is None:
        return 0.0
    data = judge(_CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context))
    return _clamp01(data.get("relevance"))


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """One evaluation example.

    ``gold_units`` enables the deterministic retrieval metrics; ``answer`` and
    ``context`` enable the LLM-judged metrics. Any subset may be provided.
    """

    question: str
    gold_units: list[str] = field(default_factory=list)
    ranked_units: list[str] = field(default_factory=list)
    context: str = ""
    answer: str = ""


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_dataset(
    cases: Sequence[EvalCase],
    *,
    k: int = 5,
    judge: Judge | None = None,
    settings: Settings | None = None,
    judged: bool = True,
) -> dict:
    """Score a dataset and return per-case rows plus aggregate means.

    Retrieval metrics are computed whenever ``gold_units`` and ``ranked_units``
    are present. LLM-judged metrics (faithfulness, answer/context relevance)
    are computed when ``judged`` is true and a judge is available; otherwise
    they are omitted so deterministic evaluation still works fully offline.
    """
    resolved_judge = _resolve_judge(judge, settings) if judged else None

    rows: list[dict] = []
    for case in cases:
        row: dict = {"question": case.question}

        if case.gold_units and case.ranked_units:
            row["retrieval"] = score_retrieval(
                case.ranked_units, case.gold_units, k=k
            ).as_dict()

        if resolved_judge is not None:
            if case.answer:
                g = verify_grounding(case.answer, case.context, judge=resolved_judge)
                row["faithfulness"] = g.score
                row["unsupported_claims"] = g.unsupported
                row["answer_relevance"] = answer_relevance(
                    case.question, case.answer, judge=resolved_judge
                )
            if case.context:
                row["context_relevance"] = context_relevance(
                    case.question, case.context, judge=resolved_judge
                )

        rows.append(row)

    return {"cases": rows, "aggregate": _aggregate(rows), "k": k}


def _aggregate(rows: list[dict]) -> dict:
    """Mean each numeric metric across rows that reported it."""
    agg: dict[str, float] = {}

    retr_keys = ["precision", "recall", "hit", "mrr", "ndcg"]
    for key in retr_keys:
        vals = [r["retrieval"][key] for r in rows if "retrieval" in r]
        if vals:
            agg[key] = _mean(vals)

    for key in ("faithfulness", "answer_relevance", "context_relevance"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            agg[key] = _mean(vals)

    return agg
