import sqlite3
import pytest

from backend.core import database
from backend.services import chat_history as ch


@pytest.fixture
def db(tmp_path, monkeypatch):
    p = tmp_path / "laidocs.db"
    monkeypatch.setattr(database, "DB_PATH", p)
    database.init_db()
    return p


def _cols(db_path, table):
    with sqlite3.connect(str(db_path)) as c:
        return {r[1]: r for r in c.execute(f"PRAGMA table_info({table})")}


def test_chat_messages_doc_id_is_nullable_after_migration(db):
    info = _cols(db, "chat_messages")
    assert "doc_id" in info
    # PRAGMA table_info column index 3 = notnull flag; 0 means nullable
    assert info["doc_id"][3] == 0


def test_global_session_ids_do_not_collide_across_docs(db):
    # Two docs that historically each had session_id=1 must be re-numbered
    # to distinct global ids by the migration.
    with sqlite3.connect(str(db)) as c:
        c.execute(
            "INSERT INTO chat_messages (doc_id, session_id, role, content) "
            "VALUES ('docA', 1, 'user', 'qA'), ('docB', 1, 'user', 'qB')"
        )
        c.commit()
    # Re-run migrations (idempotent) — must not crash, ids stay valid.
    database.init_db()
    with sqlite3.connect(str(db)) as c:
        rows = c.execute(
            "SELECT doc_id, session_id FROM chat_messages ORDER BY doc_id"
        ).fetchall()
    assert {r[0] for r in rows} == {"docA", "docB"}
    # init_db is idempotent: running twice must not duplicate rows.
    assert len(rows) == 2


def test_session_lifecycle_is_global(db):
    assert ch.get_current_session_id() == 1  # empty DB → default 1
    ch.save_message(1, "user", "hello")
    ch.save_message(1, "assistant", "hi")
    assert ch.get_current_session_id() == 1
    new_id = ch.start_new_session()
    assert new_id == 2
    ch.save_message(new_id, "user", "second topic")

    # get_messages() returns ALL rows globally, ordered.
    allm = ch.get_messages()
    assert [m["content"] for m in allm] == ["hello", "hi", "second topic"]
    assert allm[0]["session_id"] == 1 and allm[2]["session_id"] == 2

    # per-session load
    s1 = ch.get_messages_for_session(1)
    assert [m["content"] for m in s1] == ["hello", "hi"]
    s2 = ch.get_messages_for_session(2)
    assert [m["content"] for m in s2] == ["second topic"]


def test_delete_session_global(db):
    ch.save_message(1, "user", "a")
    ch.save_message(2, "user", "b")
    ch.delete_session(2)
    assert [m["content"] for m in ch.get_messages()] == ["a"]


def test_summary_injected_per_session(db):
    ch.save_compact_summary(1, "rolling summary text")
    ch.save_message(1, "user", "follow up")
    msgs = ch.get_messages_for_session(1)
    # summary becomes a synthetic user/assistant pair, then real messages
    assert msgs[0]["role"] == "user"
    assert msgs[1] == {"role": "assistant", "content": "rolling summary text"}
    assert msgs[2] == {"role": "user", "content": "follow up"}
