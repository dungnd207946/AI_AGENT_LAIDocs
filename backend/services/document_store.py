"""Shared write path for document content — keeps the 3 stores in sync.

Content lives in three places that must never drift:
  1. the ``.md`` file on disk (+ ``.md.meta.json`` sidecar) — via ``vault.save_document``
  2. the ``documents.content`` column in SQLite
  3. the ``documents.tree_index`` column (PageIndex), rebuilt from the new content

Both the REST update endpoint and the agent edit tools persist through here so
they cannot diverge on how a document is written.
"""

from __future__ import annotations

import json

from ..core.database import get_db
from ..core.vault import vault
from .tree_index import build_tree_index


async def rebuild_tree_index(doc_id: str, content: str) -> None:
    """Rebuild and store the PageIndex tree for *doc_id* from *content*.

    Stores NULL when the build yields nothing (e.g. document has no headings);
    the agent falls back to raw content in that case.
    """
    tree = await build_tree_index(content)
    with get_db() as conn:
        conn.execute(
            "UPDATE documents SET tree_index=? WHERE id=?",
            (json.dumps(tree, ensure_ascii=False) if tree else None, doc_id),
        )


async def persist_document_content(doc_id: str, new_content: str) -> None:
    """Write new Markdown content for *doc_id* to all three stores, in sync.

    Keeps the document's existing folder/filename/title/source metadata; only
    the body changes. The tree index is rebuilt inline (awaited) so a subsequent
    ``retrieve_context`` call sees content consistent with what was just written.

    Raises ValueError if the document does not exist.
    """
    result = vault.get_document(doc_id)
    if result is None:
        raise ValueError(f"Document not found: {doc_id}")

    _old_content, meta = result

    # Overwrite the .md file (same doc_id + filename → save_document overwrites in place)
    vault.save_document(
        folder=meta.folder,
        filename=meta.filename,
        content=new_content,
        title=meta.title,
        source_type=meta.source_type,
        original_path=meta.original_path,
        doc_id=doc_id,
    )

    with get_db() as conn:
        conn.execute(
            "UPDATE documents SET content=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (new_content, doc_id),
        )

    await rebuild_tree_index(doc_id, new_content)
