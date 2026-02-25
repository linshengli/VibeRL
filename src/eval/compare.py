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
    parser.add_argument("--test-cases", required=True, help="Path to test cases json")
    parser.add_argument("--output", required=True, help="Output comparison json")
    args = parser.parse_args()

    if len(args.models) != len(args.labels):
        raise ValueError("--models and --labels must have same length")

    with Path(args.test_cases).open("r", encoding="utf-8") as f:
        test_cases = json.load(f)

    evaluator = AgentEvaluator()
    rows = []
    for model, label in zip(args.models, args.labels):
        report = evaluator.evaluate(model, test_cases)
        rows.append(
            {
                "label": label,
                "model": model,
                "metrics": report.get("metrics", {}),
            }
        )

    out = {
        "test_cases": args.test_cases,
        "results": rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
