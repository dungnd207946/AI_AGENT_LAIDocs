"""SQLite database layer for LAIDocs.
Stores document metadata, folder tree, and tree index JSON.
"""
from __future__ import annotations
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from .config import LAIDOCS_HOME

DB_PATH = LAIDOCS_HOME / "data" / "laidocs.db"

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS folders (
    path TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    folder TEXT NOT NULL,
    filename TEXT NOT NULL,
    title TEXT,
    source_type TEXT NOT NULL CHECK(source_type IN ('file', 'url')),
    original_path TEXT,
    content TEXT,
    tree_index TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (folder) REFERENCES folders(path)
);
"""

# Migration for existing databases.
# IMPORTANT: append-only — never edit or reorder existing entries.
_MIGRATIONS = [
    "ALTER TABLE documents ADD COLUMN tree_index TEXT",
    """CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    session_id INTEGER NOT NULL DEFAULT 1,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)""",
    "CREATE INDEX IF NOT EXISTS idx_chat_messages_doc_id ON chat_messages(doc_id)",
    # Dense-retrieval vectors, one row per retrieval unit (tree node or chunk).
    # vector is raw float32 bytes; model identifies which embedding produced it
    # so a provider/model change can invalidate stale rows.
    """CREATE TABLE IF NOT EXISTS document_embeddings (
    doc_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    title TEXT,
    chunk TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (doc_id, unit_id)
)""",
    "CREATE INDEX IF NOT EXISTS idx_doc_embeddings_doc ON document_embeddings(doc_id)",
    # NOTE: the 'summary' role + nullable-doc_id table rebuild used to live here
    # as four destructive statements (CREATE _v2 / INSERT / DROP / RENAME). That
    # was a bug: _MIGRATIONS runs on EVERY startup, so on each restart it
    # recreated chat_messages_v2 (with doc_id NOT NULL) and renamed it over the
    # nullable table produced by guarded migration 0003 — reverting the schema
    # and breaking save_message() (which inserts without doc_id). The rebuild
    # now lives entirely in _GUARDED_MIGRATIONS (0003 + 0004), which run once.
]

# Guarded migrations: run exactly once, tracked in schema_migrations.
# Use this (not the _MIGRATIONS list) for destructive/idempotency-sensitive
# changes such as table rebuilds.
_GUARDED_MIGRATIONS: list[tuple[str, list[str]]] = [
    (
        "0003_global_sessions",
        [
            # Rebuild chat_messages: doc_id becomes nullable (provenance only,
            # no longer a key) and session_id is globalised so ids never
            # collide across documents.
            """CREATE TABLE chat_messages_v3 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                session_id INTEGER NOT NULL DEFAULT 1,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'summary')),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            # Globalise: rank each distinct (doc_id, session_id) pair to a new
            # contiguous id, ordered by (doc_id, session_id). Correlated
            # subquery avoids window-function portability concerns.
            """INSERT INTO chat_messages_v3 (id, doc_id, session_id, role, content, created_at)
               SELECT cm.id, cm.doc_id,
                 (SELECT COUNT(*) FROM (SELECT DISTINCT doc_id, session_id FROM chat_messages) d
                  WHERE (d.doc_id < cm.doc_id)
                     OR (d.doc_id = cm.doc_id AND d.session_id <= cm.session_id)) AS new_sid,
                 cm.role, cm.content, cm.created_at
               FROM chat_messages cm""",
            "DROP TABLE chat_messages",
            "ALTER TABLE chat_messages_v3 RENAME TO chat_messages",
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id)",
        ],
    ),
    (
        # Repair DBs corrupted by the old destructive _MIGRATIONS rebuild loop:
        # 0003 had already run (and is marked applied, so it never re-runs), yet
        # a later restart reverted doc_id back to NOT NULL. Rebuild once more
        # with doc_id nullable, preserving rows and ids. Idempotent for healthy
        # DBs (they are simply rebuilt to the identical schema).
        "0004_repair_doc_id_nullable",
        [
            """CREATE TABLE chat_messages_v4 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                session_id INTEGER NOT NULL DEFAULT 1,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'summary')),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """INSERT INTO chat_messages_v4 (id, doc_id, session_id, role, content, created_at)
               SELECT id, doc_id, session_id, role, content, created_at FROM chat_messages""",
            "DROP TABLE chat_messages",
            "ALTER TABLE chat_messages_v4 RENAME TO chat_messages",
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id)",
        ],
    ),
]


def init_db() -> None:
    """Create DB file and all tables if they do not exist yet."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        # Run lightweight migrations (ignore errors for already-applied ones)
        for migration in _MIGRATIONS:
            try:
                conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # column already exists or table already recreated
        _run_guarded_migrations(conn)
        conn.commit()


def _run_guarded_migrations(conn: sqlite3.Connection) -> None:
    """Run each guarded migration exactly once, tracked in schema_migrations."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "name TEXT PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    applied = {
        row[0]
        for row in conn.execute("SELECT name FROM schema_migrations").fetchall()
    }
    for name, statements in _GUARDED_MIGRATIONS:
        if name in applied:
            continue
        for stmt in statements:
            conn.execute(stmt)
        conn.execute("INSERT INTO schema_migrations (name) VALUES (?)", (name,))


# ---------------------------------------------------------------------------
# SQLite connection helpers
# ---------------------------------------------------------------------------


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection suitable for a single request."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()