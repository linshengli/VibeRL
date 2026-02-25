from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_FIELDS = {
    "sample_id",
    "messages",
    "market_type",
    "difficulty",
    "has_cot",
    "source_model",
    "generated_at",
}


def validate_file(path: Path) -> tuple[int, int]:
    ok = 0
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[INVALID] line {lineno}: not valid JSON")
                bad += 1
                continue

            missing = REQUIRED_FIELDS - set(item.keys())
            if missing:
                print(f"[INVALID] line {lineno}: missing fields {sorted(missing)}")
                bad += 1
                continue

            messages = item.get("messages")
            if not isinstance(messages, list) or not messages:
                print(f"[INVALID] line {lineno}: messages must be non-empty list")
                bad += 1
                continue

            ok += 1
    return ok, bad


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated SFT jsonl data")
    parser.add_argument("--data-dir", required=True, help="Directory containing samples.jsonl")
    args = parser.parse_args()

    path = Path(args.data_dir) / "samples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"samples.jsonl not found in {args.data_dir}")

    ok, bad = validate_file(path)
    print(json.dumps({"valid": ok, "invalid": bad, "file": str(path)}, ensure_ascii=False, indent=2))

    if bad > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
