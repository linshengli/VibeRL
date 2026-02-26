from __future__ import annotations

import json
from pathlib import Path

from src.rag import RAGStore, parse_uploaded_file


def test_rag_store_add_and_search(tmp_path: Path) -> None:
    store = RAGStore(str(tmp_path / "rag.json"))
    doc = store.add_document(
        file_name="notes.txt",
        source_type="document",
        import_type="document",
        segments=[{"text": "贵州茅台是白酒龙头，腾讯是互联网公司。", "metadata": {"section": 1}}],
    )
    assert doc["chunk_count"] >= 1

    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["file_name"] == "notes.txt"

    hits = store.search("茅台", top_k=3)
    assert len(hits) >= 1
    assert "茅台" in hits[0]["text"]


def test_parse_telegram_json() -> None:
    payload = {
        "name": "demo",
        "messages": [
            {"id": 1, "date": "2026-01-01", "from": "Alice", "text": "买入茅台"},
            {"id": 2, "date": "2026-01-02", "from": "Bob", "text": [{"type": "plain", "text": "减仓腾讯"}]},
        ],
    }
    source_type, segments = parse_uploaded_file(
        file_name="telegram-result.json",
        payload=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        import_type="telegram",
    )
    assert source_type == "telegram"
    assert len(segments) == 2
    assert "Alice" in segments[0]["text"]
    assert "减仓腾讯" in segments[1]["text"]


def test_parse_whatsapp_discord_slack_formats() -> None:
    source_type, wa_segments = parse_uploaded_file(
        file_name="chat.txt",
        payload="[12/01/24, 10:30:12 AM] Alice: 买入茅台\n[12/01/24, 10:31:00 AM] Bob: 先观望".encode("utf-8"),
        import_type="whatsapp",
    )
    assert source_type == "whatsapp"
    assert len(wa_segments) == 2
    assert "Alice" in wa_segments[0]["text"]

    discord_payload = json.dumps(
        {
            "messages": [
                {"timestamp": "2026-02-01T10:00:00Z", "author": {"name": "Tom"}, "content": "减仓腾讯"},
            ]
        },
        ensure_ascii=False,
    ).encode("utf-8")
    source_type, ds_segments = parse_uploaded_file(
        file_name="discord.json",
        payload=discord_payload,
        import_type="discord",
    )
    assert source_type == "discord"
    assert len(ds_segments) == 1
    assert "Tom" in ds_segments[0]["text"]

    slack_payload = json.dumps(
        [
            {"ts": "1712222222.0001", "user": "U123", "text": "注意回撤风险"},
        ],
        ensure_ascii=False,
    ).encode("utf-8")
    source_type, sl_segments = parse_uploaded_file(
        file_name="slack.json",
        payload=slack_payload,
        import_type="slack",
    )
    assert source_type == "slack"
    assert len(sl_segments) == 1
    assert "注意回撤风险" in sl_segments[0]["text"]
