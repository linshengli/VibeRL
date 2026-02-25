from __future__ import annotations

import json
import random
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from src.models.entities import SFTSample


class SFTDataGenerator:
    def __init__(self, model: str = "deepseek-chat", seed: int = 42) -> None:
        self.model = model
        self.rng = random.Random(seed)

    def generate_batch(
        self,
        prompts: List[str],
        market_distribution: Dict[str, float] = {
            "a_share": 0.4,
            "hk_share": 0.3,
            "mixed": 0.3,
        },
        batch_size: int = 10,
    ) -> List[SFTSample]:
        _ = batch_size

        markets = self._expand_market_distribution(len(prompts), market_distribution)
        samples: List[SFTSample] = []
        for prompt, market_type in zip(prompts, markets):
            messages = self._synthesize_messages(prompt, market_type)
            samples.append(
                SFTSample(
                    sample_id=str(uuid.uuid4()),
                    messages=messages,
                    market_type=market_type,
                    difficulty=self._infer_difficulty(prompt, market_type),
                    has_cot=True,
                    source_model=self.model,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                )
            )
        return samples

    def save_to_parquet(
        self,
        samples: List[SFTSample],
        output_path: str,
        split: float = 0.9,
    ) -> Tuple[str, str]:
        if not 0 < split < 1:
            raise ValueError("split must be in (0, 1)")

        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required. Install with: pip install pandas pyarrow") from exc

        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "messages": json.dumps(sample.messages, ensure_ascii=False),
                "data_source": "vibe_rl_stock",
            }
            for sample in samples
        ]
        self.rng.shuffle(rows)

        train_size = int(len(rows) * split)
        train_rows = rows[:train_size]
        val_rows = rows[train_size:]

        train_path = out_dir / "train.parquet"
        val_path = out_dir / "val.parquet"

        pd.DataFrame(train_rows).to_parquet(train_path, index=False)
        pd.DataFrame(val_rows).to_parquet(val_path, index=False)

        return str(train_path), str(val_path)

    def _expand_market_distribution(self, n: int, dist: Dict[str, float]) -> List[str]:
        keys = ["a_share", "hk_share", "mixed"]
        weights = [float(dist.get(k, 0.0)) for k in keys]
        s = sum(weights)
        if s <= 0:
            weights = [0.4, 0.3, 0.3]
            s = 1.0
        weights = [w / s for w in weights]
        return self.rng.choices(keys, weights=weights, k=n)

    def _synthesize_messages(self, prompt: str, market_type: str) -> List[dict]:
        thought = (
            "<think>先识别市场和股票代码，再调用行情与技术指标工具，最后汇总关键数值。</think>"
        )
        tool_hint = {
            "a_share": "search_stock_by_name -> get_realtime_quote -> get_technical_indicators",
            "hk_share": "search_stock_by_name -> get_realtime_quote",
            "mixed": "search_stock_by_name(多次) -> get_realtime_quote(多次)",
        }[market_type]
        final = f"根据分析，建议先关注趋势与成交量变化。工具链路: {tool_hint}。"

        return [
            {"role": "system", "content": "你是股票分析助手，擅长工具调用。"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{thought}\n{final}"},
        ]

    def _infer_difficulty(self, prompt: str, market_type: str) -> str:
        if market_type == "mixed" or "平安" in prompt:
            return "hard"
        if "技术指标" in prompt or "MACD" in prompt:
            return "medium"
        return "easy"


def dump_samples_jsonl(samples: List[SFTSample], output_file: str) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def load_samples_jsonl(path: str) -> List[SFTSample]:
    samples: List[SFTSample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            samples.append(SFTSample(**item))
    return samples
