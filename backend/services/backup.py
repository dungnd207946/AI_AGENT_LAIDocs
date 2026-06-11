"""Backup service — export/import .laidocs-backup archives."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.config import LAIDOCS_HOME
from ..core.database import (
    DB_PATH,
    cleanup_orphan_embeddings,
    get_db,
    init_db,
    invalidate_documents_embeddings,
)
from ..core.vault import VAULT_DIR, SYSTEM_DIRS

MANIFEST_NAME = "manifest.json"
FORMAT_VERSION = 1
APP_VERSION = "1.0.0"


# ── Stats helpers ──────────────────────────────────────────────────


def _count_meta_files(directory: Path) -> int:
    """Count .meta.json files in directory tree (= document count)."""
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.rglob("*.meta.json"))


def _count_chat_messages() -> int:
    if not DB_PATH.exists():
        return 0
    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def _count_folders() -> int:
    if not DB_PATH.exists():
        return 0
    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute("SELECT COUNT(*) FROM folders").fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def get_vault_stats() -> dict[str, Any]:
    """Return current vault statistics for the Data tab."""
    return {
        "folders": _count_folders(),
        "documents": _count_meta_files(VAULT_DIR),
        "chat_messages": _count_chat_messages(),
    }


# ── Manifest ───────────────────────────────────────────────────────


def _build_manifest() -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "app_version": APP_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats": get_vault_stats(),
    }


# ── Export ─────────────────────────────────────────────────────────


def export_backup(target_path: str) -> dict[str, Any]:
    """Create a .laidocs-backup archive at target_path."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    manifest = _build_manifest()

    try:
        with zipfile.ZipFile(str(target), "w", zipfile.ZIP_DEFLATED) as zf:
            # Write manifest
            zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False))

            # Add vault directory
            if VAULT_DIR.exists():
                for file_path in VAULT_DIR.rglob("*"):
                    if file_path.is_file():
                        arc_name = "vault/" + str(file_path.relative_to(VAULT_DIR))
                        zf.write(str(file_path), arc_name)

            # Add database
            if DB_PATH.exists():
                zf.write(str(DB_PATH), "data/laidocs.db")
    except Exception as e:
        if target.exists():
            target.unlink()
        raise e

    file_size = target.stat().st_size
    return {"success": True, "file_size": file_size, "stats": manifest["stats"]}


# ── Preview ────────────────────────────────────────────────────────


def preview_backup(source_path: str) -> dict[str, Any]:
    """Read manifest from a backup file without modifying data."""
    source = Path(source_path)
    if not source.exists():
        return {"valid": False, "error": "File not found"}

    try:
        with zipfile.ZipFile(str(source), "r") as zf:
            if MANIFEST_NAME not in zf.namelist():
                return {"valid": False, "error": "Invalid backup: missing manifest"}

            manifest = json.loads(zf.read(MANIFEST_NAME))

            if manifest.get("format_version", 0) > FORMAT_VERSION:
                return {
                    "valid": False,
                    "error": (
                        f"Backup version {manifest['format_version']} is newer "
                        f"than supported ({FORMAT_VERSION}). Please update LAIDocs."
                    ),
                }

            return {"valid": True, "manifest": manifest}
    except zipfile.BadZipFile:
        return {"valid": False, "error": "Corrupt or invalid backup file"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


# ── Import ─────────────────────────────────────────────────────────


def import_backup(source_path: str, mode: str) -> dict[str, Any]:
    """Import data from a backup file. mode: 'replace' or 'merge'."""
    preview = preview_backup(source_path)
    if not preview.get("valid"):
        raise ValueError(preview.get("error", "Invalid backup"))

    source = Path(source_path)
    if mode == "replace":
        return _import_replace(source)
    elif mode == "merge":
        return _import_merge(source)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def _safe_resolve(base_dir: Path, rel_path: str) -> Path:
    """Resolve a relative path against a base directory safely, preventing path traversal."""
    dest = (base_dir / rel_path).resolve()
    if not str(dest).startswith(str(base_dir.resolve())):
        raise ValueError(f"Malicious path detected: {rel_path}")
    return dest


def _extract_vault_files(zf: zipfile.ZipFile) -> None:
    """Extract vault/ entries from zip to VAULT_DIR."""
    for name in zf.namelist():
        if name.startswith("vault/") and not name.endswith("/"):
            rel = name[len("vault/"):]
            dest = _safe_resolve(VAULT_DIR, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src:
                dest.write_bytes(src.read())


def _extract_database(zf: zipfile.ZipFile) -> None:
    """Extract data/laidocs.db from zip to DB_PATH."""
    if "data/laidocs.db" in zf.namelist():
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with zf.open("data/laidocs.db") as src:
            DB_PATH.write_bytes(src.read())


def _extract_database_to_path(zf: zipfile.ZipFile, target: Path) -> bool:
    """Extract data/laidocs.db from zip to a specific path."""
    if "data/laidocs.db" not in zf.namelist():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open("data/laidocs.db") as src:
        target.write_bytes(src.read())
    return True


def _backup_database(source: Path, target: Path) -> None:
    """Copy a SQLite database using SQLite's backup API."""
    target.parent.mkdir(parents=True, exist_ok=True)

    src_conn = sqlite3.connect(str(source))
    dst_conn = sqlite3.connect(str(target))
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()


def _retry_path_action(action, *args) -> None:
    """Retry transient Windows file operations around SQLite file handles."""
    last_error: PermissionError | None = None
    for _ in range(10):
        try:
            action(*args)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.1)
    if last_error is not None:
        raise last_error


def _import_replace(source: Path) -> dict[str, Any]:
    """Replace all data with backup contents."""
    with zipfile.ZipFile(str(source), "r") as zf:
        manifest = json.loads(zf.read(MANIFEST_NAME))

        backup_vault = VAULT_DIR.with_name(VAULT_DIR.name + "_backup_tmp")
        backup_db = DB_PATH.with_name(DB_PATH.name + "_backup_tmp")
        restore_db = DB_PATH.with_name(DB_PATH.name + "_restore_tmp")

        try:
            # Move existing
            if VAULT_DIR.exists():
                _retry_path_action(VAULT_DIR.rename, backup_vault)
            if DB_PATH.exists():
                _backup_database(DB_PATH, backup_db)

            VAULT_DIR.mkdir(parents=True, exist_ok=True)

            # Extract everything
            _extract_vault_files(zf)
            if _extract_database_to_path(zf, restore_db):
                _backup_database(restore_db, DB_PATH)

            # Clean up backup
            if backup_vault.exists():
                shutil.rmtree(backup_vault)
            if backup_db.exists():
                _retry_path_action(backup_db.unlink)
            if restore_db.exists():
                _retry_path_action(restore_db.unlink)
        except Exception as e:
            # Rollback
            if VAULT_DIR.exists():
                shutil.rmtree(VAULT_DIR)

            if backup_vault.exists():
                _retry_path_action(backup_vault.rename, VAULT_DIR)
            if backup_db.exists():
                _backup_database(backup_db, DB_PATH)
            if restore_db.exists():
                _retry_path_action(restore_db.unlink)
            raise e

    # Reinitialize DB (apply any missing migrations)
    init_db()
    with get_db() as conn:
        conn.execute("DELETE FROM document_embeddings")

    # Ensure unsorted folder always exists
    (VAULT_DIR / "unsorted").mkdir(parents=True, exist_ok=True)

    return {
        "success": True,
        "mode": "replace",
        "imported": manifest.get("stats", {}),
    }


def _import_merge(source: Path) -> dict[str, Any]:
    """Merge backup data with existing data, skipping duplicate doc_ids."""
    imported_docs = 0
    skipped = 0
    imported_doc_ids: list[str] = []

    # Collect existing doc_ids from local database
    existing_ids: set[str] = set()
    try:
        with get_db() as conn:
            rows = conn.execute("SELECT id FROM documents").fetchall()
            existing_ids = {row["id"] for row in rows}
    except sqlite3.OperationalError:
        pass

    with zipfile.ZipFile(str(source), "r") as zf:
        # Build map of backup meta files
        backup_metas: dict[str, dict] = {}
        for name in zf.namelist():
            if name.startswith("vault/") and name.endswith(".meta.json"):
                try:
                    data = json.loads(zf.read(name))
                    backup_metas[name] = data
                except (json.JSONDecodeError, KeyError):
                    continue

        # Import documents that don't exist locally
        for meta_name, meta_data in backup_metas.items():
            doc_id = meta_data.get("doc_id", "")
            if doc_id in existing_ids:
                skipped += 1
                continue

            # Copy meta file
            rel = meta_name[len("vault/"):]
            dest = _safe_resolve(VAULT_DIR, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(meta_name))

            # Copy corresponding .md file
            md_name = meta_name.replace(".meta.json", "")
            if md_name in zf.namelist():
                md_rel = md_name[len("vault/"):]
                md_dest = _safe_resolve(VAULT_DIR, md_rel)
                md_dest.write_bytes(zf.read(md_name))

            imported_docs += 1
            imported_doc_ids.append(doc_id)

        # Copy missing asset files
        for name in zf.namelist():
            if name.startswith("vault/assets/") and not name.endswith("/"):
                rel = name[len("vault/"):]
                dest = _safe_resolve(VAULT_DIR, rel)
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zf.read(name))

        # Merge database records for new documents
        if "data/laidocs.db" in zf.namelist():
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                tmp.write(zf.read("data/laidocs.db"))
                tmp_path = tmp.name
            try:
                _merge_database(tmp_path, existing_ids)
            finally:
                os.unlink(tmp_path)

    # Sync vault folders to DB
    _sync_vault_folders_to_db()
    invalidate_documents_embeddings(imported_doc_ids)
    cleanup_orphan_embeddings()

    return {
        "success": True,
        "mode": "merge",
        "imported": {"documents": imported_docs, "skipped": skipped},
    }


# ── Database merge helpers ─────────────────────────────────────────


def _merge_database(backup_db_path: str, existing_ids: set[str]) -> None:
    """Merge document and chat records from backup DB for new documents only."""
    backup_conn = sqlite3.connect(backup_db_path)
    backup_conn.row_factory = sqlite3.Row

    try:
        # Merge documents table
        try:
            with get_db() as conn:
                cursor = backup_conn.execute("SELECT * FROM documents")
                docs_to_insert = []
                for row in cursor:
                    if row["id"] not in existing_ids:
                        docs_to_insert.append((
                            row["id"], row["folder"], row["filename"],
                            row["title"], row["source_type"],
                            row["original_path"], row["content"],
                            row["tree_index"], row["created_at"],
                            row["updated_at"]
                        ))
                if docs_to_insert:
                    conn.executemany(
                        """INSERT OR IGNORE INTO documents
                           (id, folder, filename, title, source_type,
                            original_path, content, tree_index,
                            created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        docs_to_insert
                    )
        except sqlite3.OperationalError:
            pass  # Table might not exist in backup

        # Merge chat messages for new documents only
        try:
            with get_db() as conn:
                cursor = backup_conn.execute("SELECT * FROM chat_messages")
                chats_to_insert = []
                for row in cursor:
                    if row["doc_id"] not in existing_ids:
                        chats_to_insert.append((
                            row["doc_id"], row["session_id"], row["role"],
                            row["content"], row["created_at"],
                        ))
                if chats_to_insert:
                    conn.executemany(
                        """INSERT INTO chat_messages
                           (doc_id, session_id, role, content, created_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        chats_to_insert
                    )
        except sqlite3.OperationalError:
            pass  # Table might not exist in backup
    finally:
        backup_conn.close()


def _sync_vault_folders_to_db() -> None:
    """Ensure all vault subdirectories are registered in the folders table."""
    from ..core.database import get_db

    with get_db() as conn:
        for item in VAULT_DIR.iterdir():
            if item.is_dir() and item.name not in SYSTEM_DIRS:
                rel = str(item.relative_to(VAULT_DIR))
                conn.execute(
                    "INSERT OR IGNORE INTO folders (path, name) VALUES (?, ?)",
                    (rel, item.name),
                )
