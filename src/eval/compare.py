from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.evaluator import AgentEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple models on test cases")
    parser.add_argument("--models", nargs="+", required=True, help="Model names/paths")
    parser.add_argument("--labels", nargs="+", required=True, help="Display labels for each model")
    parser.add_argument("--base-urls", nargs="+", default=None, help="API base URLs for each model (use 'none' for default)")
    parser.add_argument("--test-cases", required=True, help="Path to test cases json")
    parser.add_argument("--output", required=True, help="Output comparison json")
    args = parser.parse_args()

    if len(args.models) != len(args.labels):
        raise ValueError("--models and --labels must have same length")

    base_urls = args.base_urls or [None] * len(args.models)
    if len(base_urls) != len(args.models):
        raise ValueError("--base-urls must have same length as --models (use 'none' for default)")
    base_urls = [None if u == "none" else u for u in base_urls]

    with Path(args.test_cases).open("r", encoding="utf-8") as f:
        test_cases = json.load(f)

    evaluator = AgentEvaluator()
    rows = []
    for model, label, base_url in zip(args.models, args.labels, base_urls):
        print(f"Evaluating: {label} ({model}) @ {base_url or 'default'}...")
        report = evaluator.evaluate(model, test_cases, base_url=base_url)
        rows.append(
            {
                "label": label,
                "model": model,
                "base_url": base_url,
                "metrics": report.get("metrics", {}),
            }
        )
        # 打印实时结果
        print(f"  -> {json.dumps(report.get('metrics', {}), ensure_ascii=False)}")

    out = {
        "test_cases": args.test_cases,
        "results": rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== Final Comparison ===")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
