# tests/test_compact_manual.py
import asyncio
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, ".")

from backend.core.config import get_settings
from backend.core.database import init_db
from backend.services.compactor import compact_if_needed

DB_PATH = Path.home() / ".laidocs" / "data" / "laidocs.db"
DOC_ID = "9f34b22e-8329-4f52-ad13-9bad122660d8"
TEST_THRESHOLD = 400  # hạ ngưỡng để trigger compact với data hiện có


def show_db(label: str):
    with sqlite3.connect(str(DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT role, substr(content, 1, 80) FROM chat_messages WHERE doc_id = ?",
            (DOC_ID,),
        ).fetchall()
        total_chars = conn.execute(
            "SELECT SUM(LENGTH(content)) FROM chat_messages WHERE doc_id = ?",
            (DOC_ID,),
        ).fetchone()[0] or 0

    print(f"\n{'='*20} {label} {'='*20}")
    print(f"{'ROLE':<12} CONTENT (80 chars)")
    print("-" * 92)
    for role, content in rows:
        print(f"{role:<12} {content}")
    print(f"\nTổng: {len(rows)} rows | ~{total_chars // 4} tokens (ước tính)")


async def main():
    init_db()
    settings = get_settings()

    show_db("TRƯỚC KHI COMPACT")

    print(f"\nChạy compact với threshold={TEST_THRESHOLD} tokens...")
    compacted = await compact_if_needed(DOC_ID, settings, threshold=TEST_THRESHOLD)
    print(f"Compact thực hiện: {compacted}")

    show_db("SAU KHI COMPACT")


asyncio.run(main())