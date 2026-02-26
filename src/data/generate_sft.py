"""
SFT 数据生成入口脚本

支持两种模式：
  --mode real    (默认) 调用 DeepSeek API 执行真实 Agent 循环
  --mode mock    使用本地模板快速生成（用于测试流程）

支持多路并行生成（推荐）：
  --max-workers 控制并行线程数
  --rate-limit  控制每秒最大请求数（避免触发 API 限流）

用法：
  # 真实生成 - 串行模式（适合小批量或 API 限流严格时）
  python src/data/generate_sft.py \
    --output-dir data/sft \
    --num-samples 1000 \
    --mode real

  # 真实生成 - 多路并行（推荐！速度提升 5-10 倍）
  python src/data/generate_sft.py \
    --output-dir data/sft \
    --num-samples 1000 \
    --mode real \
    --max-workers 8 \
    --rate-limit 2.0

  # Mock 模式（测试流程用）
  python src/data/generate_sft.py \
    --output-dir data/sft \
    --num-samples 100 \
    --mode mock
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# =============================================================
#  扩充后的 Prompt 池（105 条）
#  覆盖：A 股基础 / 港股基础 / 混淆场景 / 多股票对比 /
#       指定代码 / 技术指标 / 基本面 / 综合分析 / 边界 case
# =============================================================

PROMPT_POOL = [
    # ---- A 股基础查询（20 条）----
    "帮我查一下茅台今天的实时行情",
    "看看贵州茅台最新价格",
    "查一下中国平安的行情",
    "平安银行今天涨了还是跌了",
    "比亚迪最近走势怎么样",
    "宁德时代现在多少钱",
    "帮我看看招商银行的实时行情",
    "中信证券今天表现如何",
    "查一下五粮液的实时价格",
    "泸州老窖今天的行情",
    "工商银行现在什么价位",
    "建设银行今天涨了吗",
    "中国中免最新行情是多少",
    "帮我查下海天味业的价格",
    "看看隆基绿能今天的走势",
    "三一重工最近表现怎么样",
    "万科A现在是什么价格",
    "查一下中国神华的行情",
    "长江电力今天的实时价格",
    "紫金矿业最近行情如何",

    # ---- 港股基础查询（15 条）----
    "帮我查一下腾讯的股价",
    "腾讯控股今天表现怎么样",
    "看看美团的最新行情",
    "小米集团现在多少钱",
    "查一下阿里巴巴港股的价格",
    "京东集团港股今天的走势",
    "快手科技最新行情",
    "网易港股今天表现如何",
    "百度港股价格查一下",
    "理想汽车港股最近走势",
    "蔚来在港股什么价位",
    "哔哩哔哩港股行情如何",
    "商汤科技最近表现怎么样",
    "中国海洋石油港股的价格",
    "汇丰控股最新行情",

    # ---- 混淆场景（核心！A/港同名）（15 条）----
    "查一下平安的股票",
    "帮我看看平安最近的表现",
    "中国平安的行情怎么样",
    "查一下阿里的股价",
    "阿里巴巴最近涨了多少",
    "帮我比较中国平安 A 股和港股的价格",
    "平安保险最新的行情",
    "查一下中芯国际的行情",
    "中芯国际 A 股和港股哪个更便宜",
    "比亚迪在 A 股和港股分别是什么价",
    "查一下海尔智家的行情",
    "青岛啤酒 A 股和港股的价格对比",
    "查一下药明康德",
    "李宁的股票是 A 股还是港股",
    "紫金矿业 A 股和港股的行情",

    # ---- 多只股票对比（10 条）----
    "对比腾讯和茅台的行情",
    "比较五粮液、泸州老窖和茅台",
    "帮我看看 BAT 三家的最新股价",
    "比较工商银行和建设银行的表现",
    "宁德时代和比亚迪哪个涨得多",
    "对比美团和京东港股的走势",
    "比较中国平安和招商银行",
    "帮我对比小米和蔚来的行情",
    "茅台和五粮液今天谁表现好",
    "看看腾讯、阿里、美团的最新行情",

    # ---- 指定代码查询（10 条）----
    "查一下 600519 的行情",
    "00700 最近怎么样",
    "帮我看 300750 的价格",
    "601318 今天表现如何",
    "000001 现在多少钱",
    "09988 港股的行情",
    "300059 最新价格",
    "002594 今天涨跌情况",
    "688981 最近走势",
    "01810 港股价格查一下",

    # ---- 技术指标查询（10 条）----
    "查一下茅台最近的技术指标",
    "帮我看 600519 的 MA5 和 MACD",
    "腾讯的 RSI 是多少",
    "查一下比亚迪的 MACD 和 KDJ",
    "宁德时代的均线走势怎么样",
    "平安银行的技术面分析",
    "五粮液最近的 RSI14 指标",
    "帮我看看招商银行的 MA20 和 MA60",
    "中信证券的技术指标分析",
    "查一下美团港股的 MACD",

    # ---- 基本面查询（10 条）----
    "茅台是什么行业的",
    "查一下腾讯的市值",
    "中国平安什么时候上市的",
    "比亚迪的公司基本信息",
    "宁德时代的市值多少亿",
    "招商银行属于什么行业",
    "帮我查一下阿里巴巴的公司信息",
    "五粮液的上市日期和市值",
    "万科的基本面怎么样",
    "中国神华是做什么的",

    # ---- 综合分析（10 条）----
    "帮我全面分析一下茅台",
    "给我一份腾讯控股的完整分析报告",
    "分析一下中国平安的价格和成交量变化",
    "查一下比亚迪的行情和技术指标",
    "帮我看看宁德时代，包括基本面和技术面",
    "招商银行值得买吗，给我看看数据",
    "综合分析一下五粮液的投资价值",
    "美团港股最近的走势和技术指标",
    "帮我分析泸州老窖的行情和基本面",
    "中信证券的全面分析",

    # ---- 边界 case（5 条）----
    "查一下 xxxxxx 的股票",
    "帮我看看苹果公司的行情",
    "特斯拉今天涨了吗",
    "查一下不存在的股票 999999",
    "帮我查一下",
]


def build_prompts(num_samples: int, seed: int = 42) -> list[str]:
    """从 PROMPT_POOL 中随机采样"""
    import random

    rng = random.Random(seed)
    return [rng.choice(PROMPT_POOL) for _ in range(num_samples)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT data samples")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--model", default="deepseek-chat", help="Source model name")
    parser.add_argument(
        "--mode",
        choices=["real", "mock"],
        default="real",
        help="real = 调 DeepSeek API; mock = 本地模板（测试流程用）",
    )
    parser.add_argument(
        "--market-dist",
        default='{"a_share": 0.4, "hk_share": 0.3, "mixed": 0.3}',
        help="JSON distribution for market types (mock 模式使用)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key (or set DEEPSEEK_API_KEY)")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="API base URL")

    # 多路并行参数
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="并行工作线程数，>1 时启用多路并行（默认 1，串行模式）",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=2.0,
        help="每秒最大请求数，用于 API 限流控制（默认 2.0）",
    )
    args = parser.parse_args()

    prompts = build_prompts(args.num_samples, seed=args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "real":
        from src.data.generator_real import RealSFTDataGenerator, dump_samples_jsonl, save_to_parquet

        generator = RealSFTDataGenerator(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            seed=args.seed,
        )

        def progress(current: int, total: int) -> None:
            if current % 10 == 0 or current == total:
                logger.info(f"Progress: {current}/{total}")

        # 根据 max-workers 选择串行或并行模式
        if args.max_workers > 1:
            logger.info(f"Using parallel mode: max_workers={args.max_workers}, rate_limit={args.rate_limit}/s")
            samples = generator.generate_batch_parallel(
                prompts,
                max_workers=args.max_workers,
                rate_limit_per_second=args.rate_limit,
                progress_callback=progress,
            )
        else:
            logger.info("Using sequential mode (max_workers=1)")
            samples = generator.generate_batch(prompts, progress_callback=progress)

        jsonl_path = out_dir / "samples.jsonl"
        dump_samples_jsonl(samples, str(jsonl_path))

        parquet_dir = out_dir / "parquet"
        train_path, val_path = save_to_parquet(samples, str(parquet_dir), split=0.9, seed=args.seed)

        logger.info(f"Generated {len(samples)} samples (real mode)")
        logger.info(f"JSONL: {jsonl_path}")
        logger.info(f"Train parquet: {train_path}")
        logger.info(f"Val parquet: {val_path}")

    else:
        # Mock 模式 —— 使用旧的 generator（测试流程用）
        from src.data.generator import SFTDataGenerator, dump_samples_jsonl

        market_dist = json.loads(args.market_dist)
        generator = SFTDataGenerator(model=args.model, seed=args.seed)
        samples = generator.generate_batch(prompts, market_distribution=market_dist, batch_size=10)

        jsonl_path = out_dir / "samples.jsonl"
        dump_samples_jsonl(samples, str(jsonl_path))

        parquet_dir = out_dir / "parquet"
        train_path, val_path = generator.save_to_parquet(samples, str(parquet_dir), split=0.9)

        logger.info(f"Generated {len(samples)} samples (mock mode)")
        logger.info(f"JSONL: {jsonl_path}")
        logger.info(f"Train parquet: {train_path}")
        logger.info(f"Val parquet: {val_path}")

    print(f"Generated {len(samples)} samples")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
