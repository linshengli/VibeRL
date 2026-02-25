from __future__ import annotations

import argparse
import json
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Dict, Iterable, Optional

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web.history_store import HistoryStore
from src.web.service import AgentQueryService

MAX_QUERY_CHARS = 4000
MAX_MODEL_CHARS = 120


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chunk_text(text: str, size: int = 24) -> Iterable[str]:
    if not text:
        return []
    return (text[i : i + size] for i in range(0, len(text), size))


def create_app(
    *,
    service: Optional[AgentQueryService] = None,
    store: Optional[HistoryStore] = None,
) -> Flask:
    base_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )

    service = service or AgentQueryService()
    store = store or HistoryStore("data/web_history.json")

    def _parse_chat_payload(payload: Any) -> tuple[Optional[Dict[str, Any]], Optional[tuple[Response, int]]]:
        if not isinstance(payload, dict):
            return None, (jsonify({"error": "invalid_request", "message": "请求体必须是 JSON object"}), 400)

        query = str(payload.get("query", "")).strip()
        model = str(payload.get("model", "")).strip() or None
        conversation_id = str(payload.get("conversation_id", "")).strip() or str(uuid.uuid4())

        if not query:
            return None, (jsonify({"error": "invalid_request", "message": "query 不能为空"}), 400)
        if len(query) > MAX_QUERY_CHARS:
            return (
                None,
                (
                    jsonify({"error": "invalid_request", "message": f"query 过长，最大 {MAX_QUERY_CHARS} 字符"}),
                    400,
                ),
            )
        if model is not None and len(model) > MAX_MODEL_CHARS:
            return (
                None,
                (
                    jsonify({"error": "invalid_request", "message": f"model 过长，最大 {MAX_MODEL_CHARS} 字符"}),
                    400,
                ),
            )

        return {
            "query": query,
            "model": model,
            "conversation_id": conversation_id,
        }, None

    def _build_base_record(query: str, model: Optional[str], conversation_id: str) -> Dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "turn_index": store.next_turn_index(conversation_id),
            "created_at": _now_iso(),
            "query": query,
            "model": model or service.default_model,
        }

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "time": _now_iso()})

    @app.get("/api/history")
    def history_list():
        try:
            limit = int(request.args.get("limit", 100))
        except (TypeError, ValueError):
            return jsonify({"error": "invalid_request", "message": "limit 必须为整数"}), 400
        limit = max(1, min(limit, 500))
        items = store.list_summaries(limit=limit)
        return jsonify({"items": items, "count": len(items)})

    @app.get("/api/history/<record_id>")
    def history_detail(record_id: str):
        item = store.get(record_id)
        if item is None:
            return jsonify({"error": "not_found", "record_id": record_id}), 404
        return jsonify(item)

    @app.get("/api/conversations")
    def conversation_list():
        try:
            limit = int(request.args.get("limit", 100))
        except (TypeError, ValueError):
            return jsonify({"error": "invalid_request", "message": "limit 必须为整数"}), 400
        limit = max(1, min(limit, 500))
        items = store.list_conversations(limit=limit)
        return jsonify({"items": items, "count": len(items)})

    @app.get("/api/conversations/<conversation_id>")
    def conversation_detail(conversation_id: str):
        turns = store.list_turns(conversation_id)
        if not turns:
            return jsonify({"error": "not_found", "conversation_id": conversation_id}), 404
        return jsonify(
            {
                "conversation_id": conversation_id,
                "turn_count": len(turns),
                "turns": turns,
            }
        )

    @app.post("/api/chat")
    def chat():
        parsed, err = _parse_chat_payload(request.get_json(silent=True))
        if err:
            return err
        assert parsed is not None

        query = parsed["query"]
        model = parsed["model"]
        conversation_id = parsed["conversation_id"]

        history = store.list_turns(conversation_id)
        base_record = _build_base_record(query, model, conversation_id)

        try:
            result = service.execute(query=query, model=model, conversation_history=history)
            record = {**base_record, **result}
            store.add(record)
            return jsonify(record)
        except Exception as exc:  # pragma: no cover
            error_record = {
                **base_record,
                "status": "error",
                "error": {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=3),
                },
            }
            store.add(error_record)
            return (
                jsonify(
                    {
                        "id": base_record["id"],
                        "conversation_id": conversation_id,
                        "status": "error",
                        "error": {
                            "type": "internal_error",
                            "message": "服务内部错误，请稍后重试",
                        },
                    }
                ),
                500,
            )

    @app.post("/api/chat/stream")
    def chat_stream():
        parsed, err = _parse_chat_payload(request.get_json(silent=True))
        if err:
            return err
        assert parsed is not None

        query = parsed["query"]
        model = parsed["model"]
        conversation_id = parsed["conversation_id"]

        history = store.list_turns(conversation_id)
        base_record = _build_base_record(query, model, conversation_id)

        def generate() -> Iterable[str]:
            queue: "Queue[tuple[str, Any]]" = Queue()

            def emit(event: Dict[str, Any]) -> None:
                queue.put(("agent_event", event))

            def worker() -> None:
                try:
                    result = service.execute(
                        query=query,
                        model=model,
                        conversation_history=history,
                        event_callback=emit,
                    )
                    queue.put(("result", result))
                except Exception as exc:  # pragma: no cover
                    queue.put(
                        (
                            "error",
                            {
                                "exc": exc,
                                "traceback": traceback.format_exc(limit=3),
                            },
                        )
                    )
                finally:
                    queue.put(("done", None))

            yield _sse(
                "meta",
                {
                    "conversation_id": conversation_id,
                    "turn_id": base_record["id"],
                    "turn_index": base_record["turn_index"],
                    "model": base_record["model"],
                },
            )

            Thread(target=worker, daemon=True).start()

            while True:
                kind, payload = queue.get()

                if kind == "agent_event":
                    yield _sse("agent_event", payload)
                    continue

                if kind == "result":
                    result = payload
                    chat_answer = str((result.get("chat") or {}).get("final_answer") or "")
                    multi_report = str((result.get("multi_agent") or {}).get("report") or "")
                    if chat_answer and multi_report and multi_report != chat_answer:
                        stream_text = f"{chat_answer}\n\n[Multi-Agent 汇总]\n{multi_report}"
                    else:
                        stream_text = chat_answer or multi_report

                    for piece in _chunk_text(stream_text, size=24):
                        yield _sse("delta", {"text": piece})

                    record = {**base_record, **result}
                    store.add(record)
                    yield _sse("record", record)
                    continue

                if kind == "error":
                    exc = payload.get("exc") if isinstance(payload, dict) else payload
                    tb = payload.get("traceback") if isinstance(payload, dict) else ""
                    error_record = {
                        **base_record,
                        "status": "error",
                        "error": {
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                            "traceback": tb,
                        },
                    }
                    store.add(error_record)
                    yield _sse(
                        "error",
                        {
                            "id": base_record["id"],
                            "conversation_id": conversation_id,
                            "status": "error",
                            "error": {
                                "type": "internal_error",
                                "message": "服务内部错误，请稍后重试",
                            },
                        },
                    )
                    continue

                if kind == "done":
                    yield _sse("done", {"ok": True})
                    break

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="VibeRL Web App")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", default="rule-based")
    parser.add_argument("--debug-proxy", default=None)
    parser.add_argument("--max-targets", type=int, default=3)
    parser.add_argument("--history-file", default="data/web_history.json")
    parser.add_argument("--history-max-records", type=int, default=1000)
    args = parser.parse_args()

    service = AgentQueryService(
        default_model=args.model,
        debug_proxy=args.debug_proxy,
        max_targets=args.max_targets,
    )
    store = HistoryStore(args.history_file, max_records=args.history_max_records)
    app = create_app(service=service, store=store)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
