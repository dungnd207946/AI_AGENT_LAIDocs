"""Knowledge graph + graph-of-thought reasoning over a document.

Phase 6 research layer. Where the hybrid retriever in ``retrieval.py`` ranks
*passages* lexically/densely, this module builds an **entity-relation graph**
from those same units and reasons over it, enabling:

- **Multi-hop graph retrieval**: starting from the entities named in a
  question, walk the graph a few hops and surface every source unit that
  mentions a reached entity — so a question whose answer is split across
  sections ("X was founded by Y, who also created Z") can pull all the
  relevant passages even when no single passage matches lexically.
- **Graph-of-thought**: enumerate the relation paths connecting the
  question's entities and render them as explicit reasoning chains, which can
  be handed to the agent as structured scaffolding alongside the prose context.

Design mirrors the rest of the AI layer:
- The deterministic core (graph build, k-hop subgraph, path finding,
  unit mapping, GoT rendering) has **no LLM dependency** and is fully unit
  tested offline.
- Entity/relation extraction and query-entity detection go through an
  **injectable ``extractor`` callable** so tests run without a model; in
  production omit it and a provider-agnostic extractor backed by
  ``llm.create_chat_model`` is built from ``Settings``.

Nothing here touches the frontend or the FastAPI request path; it is a library
plus a ``graph_augmented_units`` entry point that returns ranked ``unit_id``s
ready to fuse into the existing RRF pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..core.config import Settings, get_settings
from .llm import create_chat_model, is_llm_configured
from . import retrieval

logger = logging.getLogger(__name__)

# An extractor takes a single prompt string and returns the parsed JSON object
# the prompt asked for. Injectable so tests can supply a deterministic stub.
Extractor = Callable[[str], dict]

# Bounds so a pathological document/question can't blow up graph work.
MAX_HOPS = 2
MAX_TRIPLES_PER_UNIT = 30
MAX_PATHS = 20
MAX_PATH_LEN = 4  # edges


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Triple:
    """A (subject, relation, object) fact and the unit it came from."""

    subject: str
    relation: str
    obj: str
    unit_id: str = ""

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.obj)


@dataclass
class KnowledgeGraph:
    """Directed multigraph of normalized entities and labelled relations.

    Entity keys are normalized (lowercased, collapsed whitespace) for matching;
    ``labels`` keeps a human-readable surface form for display. ``unit_index``
    maps each entity to the set of source unit ids that mention it — the bridge
    back to passage retrieval.
    """

    # entity -> list of (relation, target_entity, unit_id)
    adjacency: dict[str, list[tuple[str, str, str]]] = field(default_factory=lambda: defaultdict(list))
    # entity -> set of unit ids mentioning it
    unit_index: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # normalized entity -> display label
    labels: dict[str, str] = field(default_factory=dict)

    def entities(self) -> list[str]:
        return list(self.labels.keys())

    def neighbors(self, entity: str) -> list[tuple[str, str, str]]:
        """Outgoing (relation, target, unit_id) edges for a normalized entity."""
        return self.adjacency.get(_normalize(entity), [])

    def label(self, entity: str) -> str:
        return self.labels.get(_normalize(entity), entity)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _normalize(entity: str) -> str:
    """Canonical key for entity matching: lowercased, whitespace-collapsed."""
    return _WS_RE.sub(" ", (entity or "").strip().lower())


# ---------------------------------------------------------------------------
# Triple extraction (injectable LLM, offline-testable)
# ---------------------------------------------------------------------------

_TRIPLE_PROMPT = """\
Extract factual relationships from the TEXT as (subject, relation, object) \
triples. Use short relation phrases (e.g. "founded by", "is a", "located in", \
"causes"). Only extract relations explicitly stated in the text. Resolve \
pronouns to the entity they refer to.

TEXT:
{text}

Respond with ONLY a JSON object:
{{"triples": [{{"subject": "...", "relation": "...", "object": "..."}}, ...]}}"""

_QUERY_ENTITY_PROMPT = """\
List the named entities or key noun-phrase concepts in the QUESTION that one \
would look up to answer it. Keep them short.

QUESTION:
{question}

Respond with ONLY a JSON object:
{{"entities": ["...", "..."]}}"""


def make_llm_extractor(settings: Settings, *, temperature: float = 0.0,
                       max_tokens: int = 800) -> Extractor:
    """Build a JSON-returning extractor backed by the configured chat model."""
    model = create_chat_model(settings.active_llm, temperature=temperature,
                              max_tokens=max_tokens)

    def _extract(prompt: str) -> dict:
        try:
            resp = model.invoke([{"role": "user", "content": prompt}])
            raw = resp.content if isinstance(resp.content, str) else str(resp.content)
            match = re.search(r"\{.*\}", raw or "", re.DOTALL)
            if not match:
                return {}
            data = json.loads(match.group())
            return data if isinstance(data, dict) else {}
        except Exception:
            logger.exception("KG extractor call failed")
            return {}

    return _extract


def _resolve_extractor(extractor: Extractor | None,
                       settings: Settings | None) -> Extractor | None:
    if extractor is not None:
        return extractor
    settings = settings or get_settings()
    if not is_llm_configured(settings.active_llm):
        return None
    return make_llm_extractor(settings)


def extract_triples_from_units(units: Sequence[dict], extractor: Extractor) -> list[Triple]:
    """Extract triples from each unit, tagging them with the source unit_id."""
    triples: list[Triple] = []
    for u in units:
        text = f"{u.get('title', '')}\n{u.get('text', '')}".strip()
        if not text:
            continue
        data = extractor(_TRIPLE_PROMPT.format(text=text[:6000]))
        raw = data.get("triples") or []
        for t in raw[:MAX_TRIPLES_PER_UNIT]:
            if not isinstance(t, dict):
                continue
            subj = str(t.get("subject", "")).strip()
            rel = str(t.get("relation", "")).strip()
            obj = str(t.get("object", "")).strip()
            if subj and rel and obj:
                triples.append(Triple(subj, rel, obj, str(u.get("unit_id", ""))))
    return triples


def extract_query_entities(question: str, extractor: Extractor) -> list[str]:
    """Extract the lookup entities/concepts from a question."""
    data = extractor(_QUERY_ENTITY_PROMPT.format(question=question))
    ents = data.get("entities") or []
    return [str(e).strip() for e in ents if str(e).strip()]


# ---------------------------------------------------------------------------
# Graph construction (deterministic)
# ---------------------------------------------------------------------------


def build_graph(triples: Sequence[Triple]) -> KnowledgeGraph:
    """Assemble a KnowledgeGraph from triples (deterministic, no LLM)."""
    g = KnowledgeGraph()
    for t in triples:
        s, o = _normalize(t.subject), _normalize(t.obj)
        if not s or not o:
            continue
        g.labels.setdefault(s, t.subject.strip())
        g.labels.setdefault(o, t.obj.strip())
        g.adjacency[s].append((t.relation.strip(), o, t.unit_id))
        if t.unit_id:
            g.unit_index[s].add(t.unit_id)
            g.unit_index[o].add(t.unit_id)
    return g


# ---------------------------------------------------------------------------
# Graph traversal (deterministic)
# ---------------------------------------------------------------------------


def match_entities(graph: KnowledgeGraph, query_entities: Sequence[str]) -> list[str]:
    """Map query entity strings to graph entity keys.

    Tries exact normalized match first, then substring containment in either
    direction (so "OpenAI" matches a node "openai inc"). Returns normalized
    keys, de-duplicated, preserving first-seen order.
    """
    keys = list(graph.labels.keys())
    out: list[str] = []
    seen: set[str] = set()
    for q in query_entities:
        nq = _normalize(q)
        if not nq:
            continue
        if nq in graph.labels and nq not in seen:
            out.append(nq)
            seen.add(nq)
            continue
        for k in keys:
            if (nq in k or k in nq) and k not in seen:
                out.append(k)
                seen.add(k)
    return out


def k_hop_subgraph(graph: KnowledgeGraph, seeds: Sequence[str],
                   hops: int = MAX_HOPS) -> tuple[set[str], list[tuple[str, str, str, str]]]:
    """BFS up to ``hops`` edges from seed entities.

    Returns ``(reached_entities, edges)`` where each edge is
    ``(source, relation, target, unit_id)``. Reached entities include the
    seeds. Deterministic; treats edges as undirected for *reachability* but
    reports them with their stored direction.
    """
    hops = max(0, min(hops, MAX_HOPS if hops > MAX_HOPS else hops))
    seed_keys = [_normalize(s) for s in seeds if _normalize(s) in graph.labels]
    reached: set[str] = set(seed_keys)
    edges: list[tuple[str, str, str, str]] = []
    seen_edges: set[tuple[str, str, str, str]] = set()

    # Undirected neighbor view for reachability.
    incoming: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for src, adj in graph.adjacency.items():
        for rel, tgt, uid in adj:
            incoming[tgt].append((rel, src, uid))

    frontier = deque((s, 0) for s in seed_keys)
    while frontier:
        node, depth = frontier.popleft()
        if depth >= hops:
            continue
        for rel, tgt, uid in graph.adjacency.get(node, []):
            edge = (node, rel, tgt, uid)
            if edge not in seen_edges:
                seen_edges.add(edge)
                edges.append(edge)
            if tgt not in reached:
                reached.add(tgt)
                frontier.append((tgt, depth + 1))
        for rel, src, uid in incoming.get(node, []):
            edge = (src, rel, node, uid)
            if edge not in seen_edges:
                seen_edges.add(edge)
                edges.append(edge)
            if src not in reached:
                reached.add(src)
                frontier.append((src, depth + 1))

    return reached, edges


def find_paths(graph: KnowledgeGraph, start: str, end: str,
               max_len: int = MAX_PATH_LEN) -> list[list[tuple[str, str, str]]]:
    """Find relation paths from ``start`` to ``end`` (undirected reachability).

    Each path is a list of ``(source_label, relation, target_label)`` hops.
    Bounded by ``max_len`` edges and ``MAX_PATHS`` results. Deterministic DFS.
    """
    s, e = _normalize(start), _normalize(end)
    if s not in graph.labels or e not in graph.labels:
        return []

    # Undirected adjacency with relation + direction marker for display.
    undirected: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for src, adj in graph.adjacency.items():
        for rel, tgt, _uid in adj:
            undirected[src].append((rel, tgt))
            undirected[tgt].append((f"{rel} (inv)", src))

    paths: list[list[tuple[str, str, str]]] = []

    def dfs(node: str, target: str, visited: set[str],
            acc: list[tuple[str, str, str]]):
        if len(paths) >= MAX_PATHS or len(acc) >= max_len:
            return
        for rel, nxt in undirected.get(node, []):
            if nxt in visited:
                continue
            hop = (graph.labels.get(node, node), rel, graph.labels.get(nxt, nxt))
            if nxt == target:
                paths.append(acc + [hop])
                if len(paths) >= MAX_PATHS:
                    return
                continue
            dfs(nxt, target, visited | {nxt}, acc + [hop])

    dfs(s, e, {s}, [])
    return paths


# ---------------------------------------------------------------------------
# Unit mapping & graph-augmented retrieval
# ---------------------------------------------------------------------------


def units_for_entities(graph: KnowledgeGraph, entities: Sequence[str]) -> list[str]:
    """Source unit ids mentioning any of the given entities (order-stable)."""
    out: list[str] = []
    seen: set[str] = set()
    for ent in entities:
        for uid in sorted(graph.unit_index.get(_normalize(ent), set())):
            if uid and uid not in seen:
                out.append(uid)
                seen.add(uid)
    return out


def graph_augmented_units(
    doc_id: str,
    question: str,
    settings: Settings | None = None,
    *,
    extractor: Extractor | None = None,
    units: list[dict] | None = None,
    hops: int = MAX_HOPS,
) -> list[str]:
    """Return unit ids reached by graph walk from the question's entities.

    Builds the document KG (lazily, at query time), finds the question's
    entities in it, walks ``hops`` edges, and returns the source units of all
    reached entities — ordered by hop distance (closer first). This list is
    designed to be fused as one more ranked list into ``retrieval.rrf_fuse``.

    Returns ``[]`` when no extractor is available or nothing connects, so it
    degrades to a no-op that never harms the existing pipeline.
    """
    extractor = _resolve_extractor(extractor, settings)
    if extractor is None:
        return []

    units = units if units is not None else retrieval.get_retrieval_units(doc_id)
    if not units:
        return []

    triples = extract_triples_from_units(units, extractor)
    if not triples:
        return []
    graph = build_graph(triples)

    q_entities = extract_query_entities(question, extractor)
    seeds = match_entities(graph, q_entities)
    if not seeds:
        return []

    reached, _edges = k_hop_subgraph(graph, seeds, hops=hops)
    # Seeds' own units first, then the rest of the reached frontier.
    ordered = list(seeds) + [e for e in reached if e not in set(seeds)]
    return units_for_entities(graph, ordered)


# ---------------------------------------------------------------------------
# Graph-of-thought rendering
# ---------------------------------------------------------------------------


def reason_paths(graph: KnowledgeGraph, entities: Sequence[str],
                 max_len: int = MAX_PATH_LEN) -> list[list[tuple[str, str, str]]]:
    """All relation paths between every pair of the given entities.

    The graph-of-thought primitive: it surfaces *how* the question's concepts
    connect, not just that they co-occur. Deduplicated, bounded by MAX_PATHS.
    """
    keys = match_entities(graph, entities)
    paths: list[list[tuple[str, str, str]]] = []
    seen: set[tuple] = set()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for p in find_paths(graph, keys[i], keys[j], max_len=max_len):
                sig = tuple(p)
                if sig not in seen:
                    seen.add(sig)
                    paths.append(p)
                if len(paths) >= MAX_PATHS:
                    return paths
    return paths


def render_reasoning(paths: Sequence[Sequence[tuple[str, str, str]]]) -> str:
    """Render relation paths as human-/LLM-readable reasoning chains."""
    if not paths:
        return ""
    lines = ["Reasoning paths from the knowledge graph:"]
    for idx, path in enumerate(paths, 1):
        chain = " -> ".join(
            f"{src} --[{rel}]--> {tgt}" if i == 0 else f"--[{rel}]--> {tgt}"
            for i, (src, rel, tgt) in enumerate(path)
        )
        lines.append(f"{idx}. {chain}")
    return "\n".join(lines)


def graph_of_thought(
    doc_id: str,
    question: str,
    settings: Settings | None = None,
    *,
    extractor: Extractor | None = None,
    units: list[dict] | None = None,
    max_len: int = MAX_PATH_LEN,
) -> str:
    """End-to-end graph-of-thought: build KG, connect question entities, render.

    Returns a reasoning-path scaffold string (empty when unavailable), suitable
    for prepending to the retrieved prose context handed to the agent.
    """
    extractor = _resolve_extractor(extractor, settings)
    if extractor is None:
        return ""

    units = units if units is not None else retrieval.get_retrieval_units(doc_id)
    if not units:
        return ""

    triples = extract_triples_from_units(units, extractor)
    if not triples:
        return ""
    graph = build_graph(triples)

    q_entities = extract_query_entities(question, extractor)
    paths = reason_paths(graph, q_entities, max_len=max_len)
    return render_reasoning(paths)
