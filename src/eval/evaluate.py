from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.evaluator import AgentEvaluator


def load_test_cases(path: str) -> list[dict]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stock analysis agent")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--test-cases", required=True, help="Path to test cases json")
    parser.add_argument("--output", required=True, help="Output json path")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--api-key", default=None, help="API key (or set OPENAI_API_KEY / DEEPSEEK_API_KEY)")
    args = parser.parse_args()

    evaluator = AgentEvaluator()
    test_cases = load_test_cases(args.test_cases)
    report = evaluator.evaluate(args.model, test_cases, base_url=args.base_url, api_key=args.api_key)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report.get("metrics", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
