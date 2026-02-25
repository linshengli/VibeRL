from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def try_json_loads(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return default
