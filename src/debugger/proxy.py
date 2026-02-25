from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from flask import Flask, Response, jsonify, request


class DebugStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS debug_records (
                    record_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    request_id TEXT,
                    request_body TEXT NOT NULL,
                    response_body TEXT NOT NULL,
                    tool_calls TEXT,
                    duration_ms INTEGER NOT NULL,
                    status_code INTEGER NOT NULL,
                    error TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_debug_records_timestamp ON debug_records(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_debug_records_request_id ON debug_records(request_id)")

    def insert(
        self,
        request_body: str,
        response_body: str,
        duration_ms: int,
        status_code: int,
        request_id: Optional[str],
        tool_calls: Optional[str],
        error: Optional[str],
    ) -> str:
        record_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO debug_records (
                    record_id, timestamp, request_id, request_body, response_body,
                    tool_calls, duration_ms, status_code, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    timestamp,
                    request_id,
                    request_body,
                    response_body,
                    tool_calls,
                    duration_ms,
                    status_code,
                    error,
                ),
            )
        return record_id

    def list_records(self, limit: int = 20, offset: int = 0) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT record_id, timestamp, request_id, duration_ms, status_code, error
                FROM debug_records
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_record(self, record_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM debug_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        return dict(row) if row else None


def _extract_request_id(payload: Dict[str, Any]) -> Optional[str]:
    rid = payload.get("id")
    return rid if isinstance(rid, str) else None


def _extract_tool_calls(payload: Dict[str, Any]) -> Optional[str]:
    try:
        choices = payload.get("choices", [])
        if not choices:
            return None
        msg = choices[0].get("message", {})
        tool_calls = msg.get("tool_calls")
        if tool_calls is None:
            return None
        return json.dumps(tool_calls, ensure_ascii=False)
    except Exception:
        return None


def forward_request(upstream_url: str, request_json: Dict[str, Any], api_key: Optional[str]) -> tuple[int, str, Optional[str]]:
    body = json.dumps(request_json, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urlrequest.Request(upstream_url, data=body, headers=headers, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8")
            return resp.getcode(), raw, None
    except HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp else ""
        return exc.code, raw, str(exc)
    except URLError as exc:
        return 502, json.dumps({"error": str(exc)}), str(exc)


def create_proxy_app(store: DebugStore, upstream_url: str, upstream_api_key: Optional[str]) -> Flask:
    app = Flask("debug_proxy")

    def _handle_proxy() -> Response:
        payload = request.get_json(silent=True) or {}
        inbound_auth = request.headers.get("Authorization", "")
        inbound_api_key = None
        if inbound_auth.startswith("Bearer "):
            inbound_api_key = inbound_auth.split(" ", 1)[1].strip()

        api_key = inbound_api_key or upstream_api_key
        start = time.perf_counter()
        status, response_text, err = forward_request(upstream_url, payload, api_key)
        duration_ms = int((time.perf_counter() - start) * 1000)

        parsed: Dict[str, Any] = {}
        try:
            parsed = json.loads(response_text) if response_text else {}
        except Exception:
            parsed = {}

        record_id = store.insert(
            request_body=json.dumps(payload, ensure_ascii=False),
            response_body=response_text,
            duration_ms=duration_ms,
            status_code=status,
            request_id=_extract_request_id(parsed),
            tool_calls=_extract_tool_calls(parsed),
            error=err,
        )

        headers = {"X-Debug-Record-Id": record_id}
        return Response(response=response_text, status=status, headers=headers, mimetype="application/json")

    @app.post("/proxy")
    def proxy_route() -> Response:
        return _handle_proxy()

    @app.post("/chat/completions")
    def chat_completions_route() -> Response:
        return _handle_proxy()

    @app.post("/v1/chat/completions")
    def chat_completions_v1_route() -> Response:
        return _handle_proxy()

    @app.get("/health")
    def health() -> Response:
        return jsonify({"ok": True, "upstream_url": upstream_url})

    return app


def create_ui_app(store: DebugStore) -> Flask:
    app = Flask("debug_ui")

    @app.get("/records")
    def list_records() -> Response:
        limit = int(request.args.get("limit", 20))
        offset = int(request.args.get("offset", 0))
        return jsonify({"records": store.list_records(limit=limit, offset=offset), "limit": limit, "offset": offset})

    @app.get("/records/<record_id>")
    def get_record(record_id: str) -> Response:
        record = store.get_record(record_id)
        if not record:
            return jsonify({"error": "not_found", "record_id": record_id}), 404
        return jsonify(record)

    return app


def _run_flask(app: Flask, host: str, port: int) -> None:
    app.run(host=host, port=port, debug=False, use_reloader=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP debug proxy for LLM API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080, help="Proxy port")
    parser.add_argument("--ui-port", type=int, default=8081, help="Record API/UI port")
    parser.add_argument("--db-path", default="debug/debug_records.db")
    parser.add_argument(
        "--upstream-chat-url",
        default=os.getenv("UPSTREAM_CHAT_URL", "https://api.openai.com/v1/chat/completions"),
        help="Upstream chat completion endpoint",
    )
    args = parser.parse_args()

    upstream_api_key = os.getenv("UPSTREAM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    store = DebugStore(args.db_path)

    ui_app = create_ui_app(store)
    ui_thread = threading.Thread(target=_run_flask, args=(ui_app, args.host, args.ui_port), daemon=True)
    ui_thread.start()

    proxy_app = create_proxy_app(store, args.upstream_chat_url, upstream_api_key)
    _run_flask(proxy_app, args.host, args.port)


if __name__ == "__main__":
    main()
