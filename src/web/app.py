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

from src.rag import RAGStore, parse_uploaded_file
from src.debugger.proxy import DebugStore
from src.web.history_store import HistoryStore
from src.web.service import AgentQueryService

MAX_QUERY_CHARS = 4000
MAX_MODEL_CHARS = 120
MAX_UPLOAD_MB = 32


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chunk_text(text: str, size: int = 24) -> Iterable[str]:
    if not text:
        return []
    return (text[i : i + size] for i in range(0, len(text), size))


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def create_app(
    *,
    service: Optional[AgentQueryService] = None,
    store: Optional[HistoryStore] = None,
    rag_store: Optional[RAGStore] = None,
    debug_store: Optional[DebugStore] = None,
) -> Flask:
    base_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    service = service or AgentQueryService()
    store = store or HistoryStore("data/web_history.json")
    rag_store = rag_store or RAGStore("data/rag_store.json")
    debug_store = debug_store or DebugStore("debug/debug_records.db")

    def _parse_chat_payload(payload: Any) -> tuple[Optional[Dict[str, Any]], Optional[tuple[Response, int]]]:
        if not isinstance(payload, dict):
            return None, (jsonify({"error": "invalid_request", "message": "请求体必须是 JSON object"}), 400)

        query = str(payload.get("query", "")).strip()
        model = str(payload.get("model", "")).strip() or None
        conversation_id = str(payload.get("conversation_id", "")).strip() or str(uuid.uuid4())
        use_rag = _as_bool(payload.get("use_rag"), default=True)

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
            "use_rag": use_rag,
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

    @app.get("/api/rag/docs")
    def rag_docs():
        try:
            limit = int(request.args.get("limit", 100))
        except (TypeError, ValueError):
            return jsonify({"error": "invalid_request", "message": "limit 必须为整数"}), 400
        limit = max(1, min(limit, 500))
        items = rag_store.list_documents(limit=limit)
        return jsonify({"items": items, "count": len(items)})

    @app.post("/api/rag/query")
    def rag_query():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "invalid_request", "message": "请求体必须是 JSON object"}), 400
        query = str(payload.get("query", "")).strip()
        if not query:
            return jsonify({"error": "invalid_request", "message": "query 不能为空"}), 400
        try:
            top_k = int(payload.get("top_k", 5))
        except (TypeError, ValueError):
            return jsonify({"error": "invalid_request", "message": "top_k 必须为整数"}), 400
        source_type = str(payload.get("source_type", "")).strip() or None
        hits = rag_store.search(query=query, top_k=max(1, min(top_k, 20)), source_type=source_type)
        return jsonify({"query": query, "hits": hits, "count": len(hits)})

    @app.post("/api/rag/upload")
    def rag_upload():
        upload = request.files.get("file")
        if upload is None:
            return jsonify({"error": "invalid_request", "message": "缺少 file"}), 400
        file_name = str(upload.filename or "").strip()
        if not file_name:
            return jsonify({"error": "invalid_request", "message": "文件名不能为空"}), 400

        import_type = str(request.form.get("import_type", "document")).strip().lower()
        if import_type not in {
            "document",
            "telegram",
            "telegram-chat",
            "telegram_chat",
            "chat",
            "third-party-chat",
            "third_party_chat",
            "whatsapp",
            "whatsapp-chat",
            "whatsapp_chat",
            "discord",
            "discord-chat",
            "discord_chat",
            "slack",
            "slack-chat",
            "slack_chat",
        }:
            return jsonify({"error": "invalid_request", "message": "不支持的 import_type"}), 400

        payload = upload.read()
        if not payload:
            return jsonify({"error": "invalid_request", "message": "文件为空"}), 400
        if len(payload) > MAX_UPLOAD_MB * 1024 * 1024:
            return jsonify({"error": "invalid_request", "message": f"文件过大，最大 {MAX_UPLOAD_MB}MB"}), 400

        try:
            source_type, segments = parse_uploaded_file(file_name=file_name, payload=payload, import_type=import_type)
            doc = rag_store.add_document(
                file_name=file_name,
                source_type=source_type,
                import_type=import_type,
                segments=segments,
                metadata={"content_type": upload.content_type or ""},
            )
        except Exception as exc:
            return jsonify({"error": "ingest_failed", "message": str(exc)}), 400

        return jsonify({"status": "ok", "document": doc})

    @app.get("/api/debug/records")
    def debug_records():
        try:
            limit = int(request.args.get("limit", 50))
            offset = int(request.args.get("offset", 0))
        except (TypeError, ValueError):
            return jsonify({"error": "invalid_request", "message": "limit/offset 必须为整数"}), 400
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        records = debug_store.list_records(limit=limit, offset=offset)
        return jsonify({"records": records, "limit": limit, "offset": offset, "count": len(records)})

    @app.get("/api/debug/records/<record_id>")
    def debug_record_detail(record_id: str):
        item = debug_store.get_record(record_id)
        if item is None:
            return jsonify({"error": "not_found", "record_id": record_id}), 404
        return jsonify(item)

    @app.post("/api/chat")
    def chat():
        parsed, err = _parse_chat_payload(request.get_json(silent=True))
        if err:
            return err
        assert parsed is not None

        query = parsed["query"]
        model = parsed["model"]
        conversation_id = parsed["conversation_id"]
        use_rag = parsed["use_rag"]

        history = store.list_turns(conversation_id)
        base_record = _build_base_record(query, model, conversation_id)
        rag_hits = rag_store.search(query=query, top_k=5) if use_rag else []
        rag_payload = {"enabled": use_rag, "hits": rag_hits}

        try:
            result = service.execute(
                query=query,
                model=model,
                conversation_history=history,
                rag_chunks=rag_hits,
            )
            record = {**base_record, **result, "rag": rag_payload}
            store.add(record)
            return jsonify(record)
        except Exception as exc:  # pragma: no cover
            error_record = {
                **base_record,
                "status": "error",
                "rag": rag_payload,
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
        use_rag = parsed["use_rag"]

        history = store.list_turns(conversation_id)
        base_record = _build_base_record(query, model, conversation_id)
        rag_hits = rag_store.search(query=query, top_k=5) if use_rag else []
        rag_payload = {"enabled": use_rag, "hits": rag_hits}

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
                        rag_chunks=rag_hits,
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
            yield _sse("rag_context", rag_payload)

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

                    record = {**base_record, **result, "rag": rag_payload}
                    store.add(record)
                    yield _sse("record", record)
                    continue

                if kind == "error":
                    exc = payload.get("exc") if isinstance(payload, dict) else payload
                    tb = payload.get("traceback") if isinstance(payload, dict) else ""
                    error_record = {
                        **base_record,
                        "status": "error",
                        "rag": rag_payload,
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
    parser.add_argument("--rag-file", default="data/rag_store.json")
    parser.add_argument("--rag-max-docs", type=int, default=2000)
    parser.add_argument("--rag-max-chunks", type=int, default=30000)
    parser.add_argument("--debug-db-path", default="debug/debug_records.db")
    args = parser.parse_args()

    service = AgentQueryService(
        default_model=args.model,
        debug_proxy=args.debug_proxy,
        max_targets=args.max_targets,
    )
    store = HistoryStore(args.history_file, max_records=args.history_max_records)
    rag_store = RAGStore(args.rag_file, max_documents=args.rag_max_docs, max_chunks=args.rag_max_chunks)
    debug_store = DebugStore(args.debug_db_path)
    app = create_app(service=service, store=store, rag_store=rag_store, debug_store=debug_store)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
