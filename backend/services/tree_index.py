"""Hierarchical tree index builder for markdown documents.

Adapted from PageIndex (https://github.com/VectifyAI/PageIndex).
Builds a table-of-contents-like tree from markdown headings, with
LLM-generated summaries per node for reasoning-based retrieval.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from ..core.config import Settings, get_settings


# ---------------------------------------------------------------------------
# Markdown parsing (from PageIndex page_index_md.py)
# ---------------------------------------------------------------------------


def extract_nodes_from_markdown(markdown_content: str) -> tuple[list[dict], list[str]]:
    """Parse heading nodes from markdown, respecting code blocks."""
    header_pattern = r'^(#{1,6})\s+(.+)$'
    code_block_pattern = r'^```'
    node_list = []
    lines = markdown_content.split('\n')
    in_code_block = False

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(code_block_pattern, stripped):
            in_code_block = not in_code_block
            continue
        if not stripped:
            continue
        if not in_code_block:
            match = re.match(header_pattern, stripped)
            if match:
                node_list.append({'node_title': match.group(2).strip(), 'line_num': line_num})

    return node_list, lines


def extract_node_text_content(node_list: list[dict], markdown_lines: list[str]) -> list[dict]:
    """Attach text content and heading level to each node."""
    all_nodes = []
    for node in node_list:
        line_content = markdown_lines[node['line_num'] - 1]
        header_match = re.match(r'^(#{1,6})', line_content)
        if header_match is None:
            continue
        all_nodes.append({
            'title': node['node_title'],
            'line_num': node['line_num'],
            'level': len(header_match.group(1)),
        })

    for i, node in enumerate(all_nodes):
        start_line = node['line_num'] - 1
        end_line = all_nodes[i + 1]['line_num'] - 1 if i + 1 < len(all_nodes) else len(markdown_lines)
        node['text'] = '\n'.join(markdown_lines[start_line:end_line]).strip()

    return all_nodes


def build_tree_from_nodes(node_list: list[dict]) -> list[dict]:
    """Convert flat node list into nested tree using heading levels."""
    if not node_list:
        return []
    stack: list[tuple[dict, int]] = []
    root_nodes: list[dict] = []
    node_counter = 1

    for node in node_list:
        current_level = node['level']
        tree_node = {
            'title': node['title'],
            'node_id': str(node_counter).zfill(4),
            'text': node['text'],
            'line_num': node['line_num'],
            'nodes': [],
        }
        node_counter += 1
        while stack and stack[-1][1] >= current_level:
            stack.pop()
        if not stack:
            root_nodes.append(tree_node)
        else:
            stack[-1][0]['nodes'].append(tree_node)
        stack.append((tree_node, current_level))

    return root_nodes


# ---------------------------------------------------------------------------
# Tree utilities (from PageIndex utils.py)
# ---------------------------------------------------------------------------


def structure_to_list(structure: Any) -> list[dict]:
    """Flatten tree into a list of all nodes."""
    if isinstance(structure, dict):
        nodes = [structure]
        if 'nodes' in structure:
            nodes.extend(structure_to_list(structure['nodes']))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes
    return []


def remove_fields(data: Any, fields: list[str] | None = None) -> Any:
    """Recursively remove specified fields from tree structure."""
    if fields is None:
        fields = ['text']
    if isinstance(data, dict):
        return {k: remove_fields(v, fields) for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data


def find_nodes_by_ids(structure: list[dict], node_ids: list[str]) -> list[dict]:
    """Find and return nodes matching the given node_ids."""
    id_set = set(node_ids)
    results = []

    def _traverse(nodes: list[dict]) -> None:
        for node in nodes:
            if node.get('node_id') in id_set:
                results.append(node)
            if node.get('nodes'):
                _traverse(node['nodes'])

    _traverse(structure)
    # Return in the order requested
    id_order = {nid: i for i, nid in enumerate(node_ids)}
    results.sort(key=lambda n: id_order.get(n.get('node_id', ''), 999))
    return results


# ---------------------------------------------------------------------------
# LLM summary generation
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT = """\
You are given a part of a document. Generate a concise 1-2 sentence \
description of what this section covers.

Section text:
{text}

Return only the description, nothing else."""


async def _generate_summary(text: str, settings: Settings) -> str:
    """Generate a summary for a single node using the configured LLM."""
    from .llm import create_chat_model, is_llm_configured

    # Use active_llm so env-based default credentials are honoured (previously
    # this read settings.llm directly and silently fell back to truncation when
    # only the DEFAULT_LLM_* env vars were set).
    cfg = settings.active_llm
    if not is_llm_configured(cfg):
        # LLM not configured — use truncated text as summary
        return text[:200] + ("..." if len(text) > 200 else "")

    def _call() -> str:
        model = create_chat_model(cfg, temperature=0, max_tokens=150)
        resp = model.invoke(
            [{"role": "user", "content": _SUMMARY_PROMPT.format(text=text[:3000])}]
        )
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        return content or text[:200]

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _call)
    except Exception:
        return text[:200] + ("..." if len(text) > 200 else "")


async def generate_summaries(structure: list[dict], settings: Settings) -> list[dict]:
    """Generate summaries for all nodes in the tree concurrently.

    Uses a semaphore to limit concurrent LLM requests and avoid
    rate-limit (429) errors from the provider.
    """
    sem = asyncio.Semaphore(5)
    all_nodes = structure_to_list(structure)
    tasks = []

    async def _safe_generate(text: str) -> str:
        async with sem:
            return await _generate_summary(text, settings)

    for node in all_nodes:
        node_text = node.get('text', '')
        # Short text doesn't need LLM summarisation — use text directly
        if len(node_text) < 500:

            async def _passthrough(t: str = node_text) -> str:
                return t

            tasks.append(_passthrough())
        else:
            tasks.append(_safe_generate(node_text))

    summaries = await asyncio.gather(*tasks, return_exceptions=True)

    for node, summary in zip(all_nodes, summaries):
        if isinstance(summary, Exception):
            summary = node.get('text', '')[:200]
        if not node.get('nodes'):
            node['summary'] = summary
        else:
            node['prefix_summary'] = summary

    return structure


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def build_tree_index(markdown: str, settings: Settings | None = None) -> dict | None:
    """Build a hierarchical tree index from markdown content.

    Returns a dict with 'structure' and 'line_count', or None if
    the document has no headings (fallback handled by caller).
    """
    if not markdown or not markdown.strip():
        return None

    settings = settings or get_settings()
    node_list, lines = extract_nodes_from_markdown(markdown)

    if not node_list:
        # No headings — return None, caller will use full-text fallback
        return None

    nodes_with_content = extract_node_text_content(node_list, lines)
    tree_structure = build_tree_from_nodes(nodes_with_content)

    # Generate summaries
    await generate_summaries(tree_structure, settings)

    return {
        'line_count': len(lines),
        'structure': tree_structure,
    }
