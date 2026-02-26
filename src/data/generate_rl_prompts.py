"""
生成 RL 训练用的 prompt 数据集

RL (GRPO) 训练只需要 prompt，不需要 response。
模型会在线生成 group_size 个 response 并计算 reward。

用法:
  python src/data/generate_rl_prompts.py \
    --output data/rl/prompts.parquet \
    --num-prompts 500
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 复用 generate_sft.py 的 prompt 池
from src.data.generate_sft import PROMPT_POOL


# 额外的 RL prompt（更有挑战性的场景）
RL_EXTRA_PROMPTS = [
    # 需要更强推理的场景
    "帮我分析茅台和五粮液哪个更适合长期投资",
    "比较腾讯和阿里的基本面，给出投资建议",
    "查一下宁德时代，如果 RSI 超过 70 我该怎么操作",
    "中国平安 A 股和港股有价差吗，哪个更值得买",
    "帮我看看最近白酒板块的龙头股表现",

    # 复杂多步操作
    "先查茅台的基本面，再看技术指标，最后给出综合评价",
    "对比三大运营商的行情和市值",
    "帮我分析银行板块的几只股票",
    "查一下新能源汽车相关的股票行情",

    # 带噪声的查询
    "那个，就是腾讯嘛，帮我看看",
    "茅台酒的那个股票，现在啥价",
    "平安那个，保险的，不是银行的",
    "最近比较火的那个比亚迪",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RL prompt dataset")
    parser.add_argument("--output", default="data/rl/prompts.parquet", help="Output parquet path")
    parser.add_argument("--num-prompts", type=int, default=500, help="Number of prompts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("请安装 pandas: pip install pandas pyarrow")
        sys.exit(1)

    rng = random.Random(args.seed)
    all_prompts = PROMPT_POOL + RL_EXTRA_PROMPTS

    prompts = [rng.choice(all_prompts) for _ in range(args.num_prompts)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "prompt": prompts,
        "data_source": ["vibe_rl_stock"] * len(prompts),
    })
    df.to_parquet(output_path, index=False)

    print(f"Generated {len(prompts)} RL prompts")
    print(f"Saved to: {output_path}")

    # 统计分布
    from collections import Counter
    counter = Counter(prompts)
    print(f"Unique prompts: {len(counter)}")
    print(f"Most common: {counter.most_common(3)}")


if __name__ == "__main__":
    main()
