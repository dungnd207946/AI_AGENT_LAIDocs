from pathlib import Path

import backend.services.chat_history as chat_history


def test_create_markdown_export_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_history, "DOWNLOADS_DIR", tmp_path / "downloads")

    def fake_get_messages(doc_id: str):
        return [
            {"id": 1, "session_id": 1, "role": "user", "content": "Viết báo cáo về doanh số.", "created_at": "2026-06-10 10:00:00"},
            {"id": 2, "session_id": 1, "role": "assistant", "content": "Báo cáo doanh số:\n- Doanh thu tăng 10%.", "created_at": "2026-06-10 10:00:05"},
        ]

    monkeypatch.setattr(chat_history, "get_messages", fake_get_messages)

    export_path = chat_history.create_markdown_export("doc123", filename="report.md")

    assert export_path.exists()
    assert export_path.parent == tmp_path / "downloads"
    content = export_path.read_text(encoding="utf-8")
    assert "# Chat report for document doc123" in content
    assert "## User" not in content
    assert "Viết báo cáo về doanh số." not in content
    assert "## Assistant" in content
    assert "Báo cáo doanh số:" in content
