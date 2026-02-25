from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generator import SFTDataGenerator, dump_samples_jsonl

PROMPT_POOL = [
    "帮我查一下茅台今天的实时行情",
    "看看腾讯和阿里巴巴的最新走势",
    "查一下平安最近的技术指标",
    "帮我比较贵州茅台和腾讯控股的表现",
    "分析一下中国平安的价格和成交量变化",
    "查询 600519 的 MA5 和 MACD",
    "查一下 00700 的实时价格",
]


def build_prompts(num_samples: int) -> list[str]:
    prompts = []
    for idx in range(num_samples):
        prompts.append(PROMPT_POOL[idx % len(PROMPT_POOL)])
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT data samples")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--model", default="deepseek-chat", help="Source model name")
    parser.add_argument(
        "--market-dist",
        default='{"a_share": 0.4, "hk_share": 0.3, "mixed": 0.3}',
        help="JSON distribution for market types",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    market_dist = json.loads(args.market_dist)
    generator = SFTDataGenerator(model=args.model, seed=args.seed)
    prompts = build_prompts(args.num_samples)

    samples = generator.generate_batch(prompts, market_distribution=market_dist, batch_size=10)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "samples.jsonl"
    dump_samples_jsonl(samples, str(jsonl_path))

    print(f"Generated {len(samples)} samples")
    print(f"Saved to: {jsonl_path}")


if __name__ == "__main__":
    main()
