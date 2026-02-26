from __future__ import annotations

import io
import json
from pathlib import Path

from src.rag.store import RAGStore
from src.web.app import create_app
from src.web.history_store import HistoryStore


class FakeService:
    def __init__(self) -> None:
        self.default_model = "rule-based"

    def execute(
        self,
        query: str,
        model: str | None = None,
        conversation_history=None,
        rag_chunks=None,
        event_callback=None,
    ):
        if query == "boom":
            raise RuntimeError("boom from fake service")
        if event_callback is not None:
            event_callback({"type": "stage", "stage": "chat.start", "detail": {"history_len": len(conversation_history or [])}})

        return {
            "status": "ok",
            "model": model or self.default_model,
            "chat": {
                "trajectory_id": "traj_1",
                "final_answer": f"answer:{query}",
                "num_turns": 2,
                "success": True,
                "tool_calls": ["search_stock_by_name", "get_realtime_quote"],
                "messages": [],
                "tool_trace": [],
            },
            "multi_agent": {
                "query": query,
                "num_targets": 1,
                "analyses": [],
                "report": f"report:{query}",
                "worklog": [
                    {
                        "agent": "PlannerAgent",
                        "title": "拆解任务",
                        "detail": {"tasks": [{"name": "茅台"}]},
                    }
                ],
            },
            "rag": {"used": len(rag_chunks or [])},
        }


def build_client(tmp_path: Path):
    store = HistoryStore(str(tmp_path / "web_history_test.json"))
    rag_store = RAGStore(str(tmp_path / "rag_store_test.json"))
    app = create_app(service=FakeService(), store=store, rag_store=rag_store)
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_and_index(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    index_resp = client.get("/")
    assert index_resp.status_code == 200
    assert "VibeRL Agent Studio" in index_resp.get_data(as_text=True)

    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    assert health_resp.get_json()["ok"] is True


def test_chat_and_history_flow(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    chat_resp = client.post("/api/chat", json={"query": "测试一下", "model": "rule-based"})
    assert chat_resp.status_code == 200
    record = chat_resp.get_json()
    assert record["status"] == "ok"
    assert record["chat"]["final_answer"] == "answer:测试一下"
    record_id = record["id"]
    conversation_id = record["conversation_id"]

    list_resp = client.get("/api/history")
    assert list_resp.status_code == 200
    items = list_resp.get_json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == record_id

    convo_list = client.get("/api/conversations")
    assert convo_list.status_code == 200
    assert convo_list.get_json()["count"] == 1
    assert convo_list.get_json()["items"][0]["conversation_id"] == conversation_id

    convo_detail = client.get(f"/api/conversations/{conversation_id}")
    assert convo_detail.status_code == 200
    assert convo_detail.get_json()["turn_count"] == 1

    detail_resp = client.get(f"/api/history/{record_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.get_json()
    assert detail["multi_agent"]["worklog"][0]["agent"] == "PlannerAgent"


def test_invalid_and_not_found(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    invalid_resp = client.post("/api/chat", json={"query": "  "})
    assert invalid_resp.status_code == 400
    assert invalid_resp.get_json()["error"] == "invalid_request"

    not_found_resp = client.get("/api/history/not-exists")
    assert not_found_resp.status_code == 404
    assert not_found_resp.get_json()["error"] == "not_found"

    invalid_limit = client.get("/api/history?limit=abc")
    assert invalid_limit.status_code == 400
    assert invalid_limit.get_json()["error"] == "invalid_request"

    too_long_query = client.post("/api/chat", json={"query": "x" * 4001})
    assert too_long_query.status_code == 400
    assert too_long_query.get_json()["error"] == "invalid_request"


def test_chat_error_recorded(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    err_resp = client.post("/api/chat", json={"query": "boom"})
    assert err_resp.status_code == 500
    payload = err_resp.get_json()
    assert payload["status"] == "error"
    assert payload["error"]["type"] == "internal_error"

    history = client.get("/api/history").get_json()["items"]
    assert len(history) == 1
    assert history[0]["status"] == "error"


def test_chat_stream_endpoint(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    resp = client.post("/api/chat/stream", json={"query": "流式测试", "model": "rule-based"})
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "event: meta" in body
    assert "event: rag_context" in body
    assert "event: agent_event" in body
    assert "event: record" in body
    assert "event: done" in body

    # Parse final record event payload
    record_payload = None
    for block in body.split("\n\n"):
        if "event: record" not in block:
            continue
        for line in block.split("\n"):
            if line.startswith("data:"):
                record_payload = json.loads(line[5:].strip())
                break
    assert record_payload is not None
    assert record_payload["status"] == "ok"


def test_rag_upload_and_query(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    upload = client.post(
        "/api/rag/upload",
        data={
            "import_type": "document",
            "file": (io.BytesIO("这是一段测试知识库内容，包含茅台和腾讯。".encode("utf-8")), "notes.txt"),
        },
        content_type="multipart/form-data",
    )
    assert upload.status_code == 200
    doc = upload.get_json()["document"]
    assert doc["chunk_count"] >= 1

    docs = client.get("/api/rag/docs")
    assert docs.status_code == 200
    assert docs.get_json()["count"] == 1

    query = client.post("/api/rag/query", json={"query": "茅台", "top_k": 3})
    assert query.status_code == 200
    assert query.get_json()["count"] >= 1

    whatsapp_upload = client.post(
        "/api/rag/upload",
        data={
            "import_type": "whatsapp",
            "file": (io.BytesIO("[12/01/24, 10:30:12 AM] Alice: 买入茅台".encode("utf-8")), "wa.txt"),
        },
        content_type="multipart/form-data",
    )
    assert whatsapp_upload.status_code == 200
    assert whatsapp_upload.get_json()["document"]["source_type"] == "whatsapp"


def test_debugger_api_available(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    records = client.get("/api/debug/records?limit=10&offset=0")
    assert records.status_code == 200
    payload = records.get_json()
    assert "records" in payload
    assert payload["count"] >= 0

    missing = client.get("/api/debug/records/not-found")
    assert missing.status_code == 404
