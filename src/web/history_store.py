from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class HistoryStore:
    def __init__(self, file_path: str, max_records: int = 1000) -> None:
        self.file_path = Path(file_path)
        self.max_records = max(1, int(max_records))
        self._lock = threading.Lock()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("[]\n", encoding="utf-8")

    def _load(self) -> List[Dict[str, Any]]:
        raw = self.file_path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Keep a backup for debugging and continue with a clean in-memory state.
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = self.file_path.with_suffix(self.file_path.suffix + f".corrupt.{ts}")
            backup.write_text(raw, encoding="utf-8")
            self.file_path.write_text("[]\n", encoding="utf-8")
            return []
        if not isinstance(data, list):
            return []
        return data

    def _save(self, items: List[Dict[str, Any]]) -> None:
        self.file_path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def add(self, record: Dict[str, Any]) -> None:
        with self._lock:
            items = self._load()
            items.append(record)
            if len(items) > self.max_records:
                items = items[-self.max_records :]
            self._save(items)

    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            items = self._load()
        for item in items:
            if item.get("id") == record_id:
                return item
        return None

    def list_summaries(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            items = self._load()

        items_sorted = sorted(items, key=lambda x: str(x.get("created_at", "")), reverse=True)
        summaries: List[Dict[str, Any]] = []
        for item in items_sorted[:limit]:
            final_answer = item.get("chat", {}).get("final_answer") if isinstance(item.get("chat"), dict) else None
            report = item.get("multi_agent", {}).get("report") if isinstance(item.get("multi_agent"), dict) else None
            preview = final_answer or report or ""
            summaries.append(
                {
                    "id": item.get("id"),
                    "created_at": item.get("created_at"),
                    "query": item.get("query"),
                    "model": item.get("model"),
                    "status": item.get("status"),
                    "preview": str(preview)[:120],
                }
            )
        return summaries

    def list_turns(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            items = self._load()
        turns = [x for x in items if x.get("conversation_id") == conversation_id]
        turns.sort(key=lambda x: str(x.get("created_at", "")))
        return turns

    def list_conversations(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            items = self._load()

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in items:
            cid = str(item.get("conversation_id") or item.get("id") or "")
            if not cid:
                continue
            grouped.setdefault(cid, []).append(item)

        out: List[Dict[str, Any]] = []
        for cid, turns in grouped.items():
            turns_sorted = sorted(turns, key=lambda x: str(x.get("created_at", "")))
            first = turns_sorted[0]
            last = turns_sorted[-1]
            out.append(
                {
                    "conversation_id": cid,
                    "created_at": first.get("created_at"),
                    "updated_at": last.get("created_at"),
                    "model": last.get("model"),
                    "turn_count": len(turns_sorted),
                    "last_query": last.get("query"),
                    "last_status": last.get("status"),
                    "last_preview": str(
                        (
                            (last.get("chat") or {}).get("final_answer")
                            or (last.get("multi_agent") or {}).get("report")
                            or ""
                        )
                    )[:120],
                }
            )

        out.sort(key=lambda x: str(x.get("updated_at", "")), reverse=True)
        return out[:limit]

    def next_turn_index(self, conversation_id: str) -> int:
        turns = self.list_turns(conversation_id)
        return len(turns) + 1
