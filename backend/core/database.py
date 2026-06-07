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

# Migration for existing databases
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
]


def init_db() -> None:
    """Create DB file and all tables if they do not exist yet."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        # Run migrations (ignore errors for already-applied ones)
        for migration in _MIGRATIONS:
            try:
                conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # column already exists


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
