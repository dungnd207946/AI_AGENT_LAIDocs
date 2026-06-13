"""Folder management API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..core.database import get_db
from ..core.vault import vault
from ..models.document import DocumentSummary, FolderCreate, FolderNode, FolderRename

router = APIRouter(prefix="/api/folders", tags=["folders"])


# ── Routes ───────────────────────────────────────────────────────


@router.get("/", response_model=list[FolderNode])
def list_folders():
    """Return a flat list of all folders with document counts."""
    folders = vault.list_folders()
    
    with get_db() as db:
        cursor = db.execute("SELECT folder, COUNT(*) as count FROM documents GROUP BY folder")
        counts = {row["folder"] or "": row["count"] for row in cursor.fetchall()}

    result: list[FolderNode] = []
    for f in folders:
        result.append(
            FolderNode(
                path=f["path"],
                name=f["name"],
                parent_path=f.get("parent_path"),
                document_count=counts.get(f["path"], 0),
            )
        )
    return result


@router.get("/tree", response_model=list[FolderNode])
def get_folder_tree():
    """Return a nested tree of all folders with their documents."""
    folders = vault.list_folders()
    
    with get_db() as db:
        cursor = db.execute("SELECT id, folder, title, filename, source_type FROM documents ORDER BY title COLLATE NOCASE")
        docs_by_folder: dict[str, list[DocumentSummary]] = {}
        for row in cursor.fetchall():
            folder_path = row["folder"] or ""
            if folder_path not in docs_by_folder:
                docs_by_folder[folder_path] = []
            docs_by_folder[folder_path].append(
                DocumentSummary(
                    id=row["id"],
                    title=row["title"] or row["filename"].removesuffix(".md"),
                    filename=row["filename"],
                    source_type=row["source_type"]
                )
            )

    # Build lookup: path -> FolderNode
    node_map: dict[str, FolderNode] = {}
    for f in folders:
        path = f["path"]
        node_map[path] = FolderNode(
            path=path,
            name=f["name"],
            parent_path=f.get("parent_path"),
            document_count=len(docs_by_folder.get(path, [])),
            documents=docs_by_folder.get(path, []),
        )

    # Build tree: attach children to parents
    roots: list[FolderNode] = []
    for path, node in node_map.items():
        parent = node.parent_path
        if parent and parent in node_map:
            node_map[parent].children.append(node)
        else:
            roots.append(node)

    # Sort: "unsorted" pinned to top, then alphabetically at each level
    def _sort_tree(nodes: list[FolderNode]) -> list[FolderNode]:
        for n in nodes:
            n.children = _sort_tree(n.children)
        return sorted(nodes, key=lambda n: (0 if n.path == "unsorted" else 1, n.name.lower()))

    return _sort_tree(roots)


@router.post("/", response_model=FolderNode, status_code=201)
def create_folder(body: FolderCreate):
    """Create a new folder on disk and register it in SQLite."""
    # Validate folder depth (max 3 levels)
    parts = Path(body.path).parts
    if len(parts) > 3:
        raise HTTPException(status_code=400, detail="Maximum folder depth is 3 levels.")
        
    # Determine parent_path from the given path
    parent = str(Path(body.path).parent) if str(Path(body.path).parent) != "." else None

    # Create on disk via vault
    try:
        info = vault.create_folder(body.path, body.name, parent_path=parent)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # Persist to SQLite
    with get_db() as db:
        db.execute(
            "INSERT OR IGNORE INTO folders (path, name, parent_path, created_at) VALUES (?, ?, ?, ?)",
            (info["path"], info["name"], info.get("parent_path"), info["created_at"]),
        )

    return FolderNode(
        path=info["path"],
        name=info["name"],
        parent_path=info.get("parent_path"),
        document_count=0,
    )


@router.put("/rename", response_model=dict)
def rename_folder(body: FolderRename):
    """Rename (move) a folder on disk and update the SQLite record."""
    try:
        info = vault.rename_folder(body.path, body.new_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    # Update SQLite — folder path
    with get_db() as db:
        db.execute("UPDATE folders SET path = ? WHERE path = ?", (info["new_path"], info["old_path"]))
        # Update any documents that reference the old folder path
        db.execute("UPDATE documents SET folder = ? WHERE folder = ?", (info["new_path"], info["old_path"]))

    return info


@router.delete("/{path:path}", status_code=204)
def delete_folder(path: str):
    """Delete a folder, its contents, and all associated SQLite records."""
    try:
        vault.delete_folder(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    with get_db() as db:
        folder_prefix = f"{path}/%"
        folder_prefix_windows = f"{path}\\%"
        rows = db.execute(
            """SELECT id FROM documents
               WHERE folder = ? OR folder LIKE ? OR folder LIKE ?""",
            (path, folder_prefix, folder_prefix_windows),
        ).fetchall()
        doc_ids = [row["id"] for row in rows]
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            db.execute(
                f"DELETE FROM document_embeddings WHERE doc_id IN ({placeholders})",
                doc_ids,
            )

        db.execute(
            "DELETE FROM documents WHERE folder = ? OR folder LIKE ? OR folder LIKE ?",
            (path, folder_prefix, folder_prefix_windows),
        )
        db.execute(
            "DELETE FROM folders WHERE path = ? OR path LIKE ? OR path LIKE ?",
            (path, folder_prefix, folder_prefix_windows),
        )
