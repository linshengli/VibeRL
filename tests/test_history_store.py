from __future__ import annotations

from pathlib import Path

from src.web.history_store import HistoryStore


def test_history_store_keeps_recent_records(tmp_path: Path) -> None:
    store = HistoryStore(str(tmp_path / "history.json"), max_records=2)
    store.add({"id": "1", "created_at": "2026-01-01T00:00:00+00:00"})
    store.add({"id": "2", "created_at": "2026-01-02T00:00:00+00:00"})
    store.add({"id": "3", "created_at": "2026-01-03T00:00:00+00:00"})

    items = store.list_summaries(limit=10)
    ids = [x["id"] for x in items]
    assert ids == ["3", "2"]


def test_history_store_conversation_views(tmp_path: Path) -> None:
    store = HistoryStore(str(tmp_path / "history_conv.json"), max_records=10)
    store.add(
        {
            "id": "r1",
            "conversation_id": "c1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "query": "q1",
            "status": "ok",
            "chat": {"final_answer": "a1"},
        }
    )
    store.add(
        {
            "id": "r2",
            "conversation_id": "c1",
            "created_at": "2026-01-01T00:01:00+00:00",
            "query": "q2",
            "status": "ok",
            "chat": {"final_answer": "a2"},
        }
    )
    store.add(
        {
            "id": "r3",
            "conversation_id": "c2",
            "created_at": "2026-01-01T00:02:00+00:00",
            "query": "q3",
            "status": "ok",
            "chat": {"final_answer": "a3"},
        }
    )

    conversations = store.list_conversations()
    assert conversations[0]["conversation_id"] == "c2"
    assert conversations[1]["conversation_id"] == "c1"
    assert conversations[1]["turn_count"] == 2

    turns_c1 = store.list_turns("c1")
    assert [t["id"] for t in turns_c1] == ["r1", "r2"]
    assert store.next_turn_index("c1") == 3


def test_history_store_handles_corrupted_file(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text("{not-json", encoding="utf-8")

    store = HistoryStore(str(path), max_records=10)
    # should not raise and should auto-reset file
    items = store.list_summaries()
    assert items == []

    backups = list(tmp_path.glob("history.json.corrupt.*"))
    assert backups, "corrupted history should be backed up"
    assert path.read_text(encoding="utf-8").strip() == "[]"
