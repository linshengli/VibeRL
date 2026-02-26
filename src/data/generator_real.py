"""
真实 SFT 数据生成器 —— 调用 DeepSeek API 执行完整 Agent 循环

与 generator.py 的区别：
- generator.py 是假的，只拼接硬编码模板
- 本文件真正调用 DeepSeek API，让大模型执行 ReAct 循环，记录含 tool_calls 的多轮对话

用法：
    from src.data.generator_real import RealSFTDataGenerator
    gen = RealSFTDataGenerator(api_key="sk-xxx")
    sample = gen.generate_one("帮我查一下茅台今天的行情")

    # 多路并行生成
    samples = gen.generate_batch_parallel(prompts, max_workers=8)
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.models.entities import SFTSample
from src.tools.stock_tools import get_stock_tool_registry, get_tool_definitions

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一个专业的股票分析助手，擅长通过工具调用来获取和分析股票数据。\n\n"
    "工作流程：\n"
    "1. 先理解用户的需求，识别涉及的股票和市场（A股/港股）\n"
    "2. 调用 search_stock_by_name 搜索股票代码\n"
    "3. 根据需求调用 get_realtime_quote / get_stock_info / get_technical_indicators\n"
    "4. 基于工具返回的数据给出清晰、有条理的分析\n\n"
    "注意：\n"
    "- A股代码为6位数字（如 600519），港股代码为5位数字（如 00700）\n"
    "- 同名股票可能同时存在于 A 股和港股（如 中国平安 601318 / 02318），注意区分\n"
    "- 每次工具调用前先思考（<think>...</think>），说明为什么要调用这个工具\n"
    "- 最终回答要包含具体数字和事实，不要空泛"
)


class RealSFTDataGenerator:
    """
    用 DeepSeek API 真正执行 Agent 循环，生成高质量多轮 tool calling 对话。

    每条数据的 messages 列表里会包含：
    - system prompt
    - user query
    - assistant (含 tool_calls)
    - tool (工具返回结果)
    - assistant (再次思考或最终回答)
    - ...循环直到最终回答
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        seed: int = 42,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or ""
        self.base_url = base_url
        self.model = model
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.tools = get_stock_tool_registry()
        self.tool_defs = get_tool_definitions()

        self._client = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError("请安装 openai: pip install openai")

            if not self.api_key:
                raise RuntimeError(
                    "请设置 DEEPSEEK_API_KEY 环境变量，或传入 api_key 参数"
                )

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate_one(
        self,
        prompt: str,
        market_type: str = "auto",
        max_turns: int = 10,
    ) -> SFTSample:
        """
        执行一次完整的 Agent ReAct 循环，返回 SFT 训练样本。

        Args:
            prompt: 用户查询
            market_type: a_share / hk_share / mixed / auto
            max_turns: 最大轮数
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        tool_call_count = 0
        has_final_answer = False

        for turn in range(max_turns):
            resp = self._call_api(messages)
            if resp is None:
                # API 调用失败，添加一个兜底回答
                messages.append({
                    "role": "assistant",
                    "content": "抱歉，暂时无法完成分析，请稍后重试。",
                })
                break

            msg = resp.choices[0].message

            if msg.tool_calls:
                # 记录 assistant 的 tool_calls
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                # 执行每个工具调用
                for tc in msg.tool_calls:
                    tool_result = self._execute_tool(tc.function.name, tc.function.arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })
                    tool_call_count += 1
            else:
                # 最终回答（没有 tool_calls）
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                })
                has_final_answer = True
                break

        # 如果循环结束还没给最终回答，强制补一个
        if not has_final_answer:
            messages.append({
                "role": "assistant",
                "content": "分析完成，但由于调用次数限制，部分信息可能不完整。",
            })

        # 推断 market_type
        if market_type == "auto":
            market_type = self._infer_market_type(messages)

        # 推断 difficulty
        difficulty = self._infer_difficulty(prompt, market_type, tool_call_count)

        return SFTSample(
            sample_id=str(uuid.uuid4()),
            messages=messages,
            market_type=market_type,
            difficulty=difficulty,
            has_cot=self._has_think_tags(messages),
            source_model=self.model,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def generate_batch(
        self,
        prompts: List[str],
        market_distribution: Optional[Dict[str, float]] = None,
        batch_size: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[SFTSample]:
        """
        批量生成 SFT 数据（串行版本）。

        Args:
            prompts: prompt 列表
            market_distribution: 忽略（为了兼容旧接口保留）
            batch_size: 忽略（为了兼容旧接口保留）
            progress_callback: 回调 fn(current, total)
        """
        samples: List[SFTSample] = []
        total = len(prompts)

        for idx, prompt in enumerate(prompts):
            try:
                sample = self.generate_one(prompt)
                samples.append(sample)
                logger.info(f"[{idx + 1}/{total}] OK: {prompt[:40]}...")
            except Exception as exc:
                logger.warning(f"[{idx + 1}/{total}] FAIL: {prompt[:40]}... => {exc}")

            if progress_callback:
                progress_callback(idx + 1, total)

            # 间隔，避免 rate limit
            if idx < total - 1:
                time.sleep(0.5)

        return samples

    def generate_batch_parallel(
        self,
        prompts: List[str],
        max_workers: int = 8,
        rate_limit_per_second: float = 2.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[SFTSample]:
        """
        批量生成 SFT 数据（多路并行版本）。

        使用线程池并行调用 DeepSeek API，大幅提升数据生成速度。
        通过 rate_limit_per_second 控制整体请求速率，避免触发 API 限流。

        Args:
            prompts: prompt 列表
            max_workers: 并行工作线程数（建议 4-16，根据 API 限流调整）
            rate_limit_per_second: 每秒最大请求数（DeepSeek 默认约 2-5 req/s）
            progress_callback: 回调 fn(current, total)

        Returns:
            生成的 SFTSample 列表（顺序与 prompts 一致，失败的返回 None）

        用法示例：
            gen = RealSFTDataGenerator(api_key="sk-xxx")
            samples = gen.generate_batch_parallel(
                prompts,
                max_workers=8,
                rate_limit_per_second=2.0
            )
        """
        import threading

        total = len(prompts)
        samples: List[Optional[SFTSample]] = [None] * total
        completed = 0
        lock = threading.Lock()
        last_request_time = [0.0]
        time_lock = threading.Lock()

        def rate_limited_generate(idx_prompt: Tuple[int, str]) -> Tuple[int, Optional[SFTSample]]:
            idx, prompt = idx_prompt

            # 速率限制：确保两次请求之间有足够的间隔
            with time_lock:
                elapsed = time.time() - last_request_time[0]
                min_interval = 1.0 / rate_limit_per_second
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                last_request_time[0] = time.time()

            try:
                sample = self.generate_one(prompt)
                logger.info(f"[{idx + 1}/{total}] OK: {prompt[:40]}...")
                return idx, sample
            except Exception as exc:
                logger.warning(f"[{idx + 1}/{total}] FAIL: {prompt[:40]}... => {exc}")
                return idx, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(rate_limited_generate, (i, p)): i
                for i, p in enumerate(prompts)
            }

            for future in as_completed(future_to_idx):
                idx, sample = future.result()
                if sample is not None:
                    samples[idx] = sample

                with lock:
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)

        # 过滤掉失败的样本
        valid_samples = [s for s in samples if s is not None]
        logger.info(f"Parallel generation completed: {len(valid_samples)}/{total} succeeded")
        return valid_samples

    def _call_api(self, messages: List[Dict[str, Any]]) -> Any:
        """调用 DeepSeek API，带重试"""
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_defs,
                    tool_choice="auto",
                    temperature=0.3,
                )
                return resp
            except Exception as exc:
                logger.warning(f"API call attempt {attempt + 1} failed: {exc}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        return None

    def _execute_tool(self, tool_name: str, arguments_json: str) -> Dict[str, Any]:
        """执行工具调用"""
        tool = self.tools.get(tool_name)
        if tool is None:
            return {"error": True, "message": f"Unknown tool: {tool_name}"}

        try:
            args = json.loads(arguments_json)
        except json.JSONDecodeError:
            return {"error": True, "message": f"Invalid JSON arguments: {arguments_json}"}

        try:
            return tool.handler(**args)
        except Exception as exc:
            return {"error": True, "message": f"Tool execution failed: {exc}"}

    def _infer_market_type(self, messages: List[Dict[str, Any]]) -> str:
        """从对话中推断 market type"""
        has_a = False
        has_hk = False
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("content"):
                content = msg["content"]
                if '"market": "a_share"' in content:
                    has_a = True
                if '"market": "hk_share"' in content:
                    has_hk = True
        if has_a and has_hk:
            return "mixed"
        if has_hk:
            return "hk_share"
        return "a_share"

    def _infer_difficulty(self, prompt: str, market_type: str, tool_count: int) -> str:
        if market_type == "mixed" or tool_count >= 6:
            return "hard"
        if tool_count >= 3 or any(k in prompt.lower() for k in ["技术指标", "macd", "对比", "比较"]):
            return "medium"
        return "easy"

    def _has_think_tags(self, messages: List[Dict[str, Any]]) -> bool:
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                if "<think>" in msg["content"]:
                    return True
        return False


def dump_samples_jsonl(samples: List[SFTSample], output_file: str) -> None:
    """保存到 JSONL 文件"""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def load_samples_jsonl(path: str) -> List[SFTSample]:
    """从 JSONL 文件加载"""
    samples: List[SFTSample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            samples.append(SFTSample(**item))
    return samples


def save_to_parquet(
    samples: List[SFTSample],
    output_dir: str,
    split: float = 0.9,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    保存为 verl 格式的 parquet 文件。

    verl SFT 需要的 parquet 格式：
    - messages: JSON string of messages list
    - data_source: string identifier
    """
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("请安装 pandas: pip install pandas pyarrow")

    rng = random.Random(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample in samples:
        rows.append({
            "messages": json.dumps(sample.messages, ensure_ascii=False),
            "data_source": "vibe_rl_stock",
        })

    rng.shuffle(rows)

    train_size = int(len(rows) * split)
    train_rows = rows[:train_size]
    val_rows = rows[train_size:]

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows or train_rows[:1]).to_parquet(val_path, index=False)

    return str(train_path), str(val_path)
