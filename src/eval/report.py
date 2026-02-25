from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown report from compare output")
    parser.add_argument("--input", required=True, help="Comparison json path")
    parser.add_argument("--output", required=True, help="Markdown output path")
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        data = json.load(f)

    lines = [
        "# 模型对比评估报告",
        "",
        f"测试集: `{data.get('test_cases', '')}`",
        "",
        "| 模型标签 | tool_accuracy | output_quality | amb_accuracy |",
        "|---|---:|---:|---:|",
    ]

    for row in data.get("results", []):
        m = row.get("metrics", {})
        lines.append(
            f"| {row.get('label', '')} | {m.get('tool_accuracy', 0):.4f} | {m.get('output_quality', 0):.4f} | {m.get('amb_accuracy', 0):.4f} |"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"report written: {out_path}")


if __name__ == "__main__":
    main()
