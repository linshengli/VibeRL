"""
LLM-as-Judge 评分器

用大模型（如 DeepSeek）对 Agent 的输出质量打分。
可以作为 reward 信号的一部分，替代纯规则评分。

用法:
    from src.reward.llm_judge import LLMJudge
    judge = LLMJudge(api_key="sk-xxx")
    score = judge.score("查一下茅台的行情", "贵州茅台(600519)当前价格1680元...")
    # score: 0.0 ~ 1.0
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = "你是一个严格的股票分析回答质量评估专家。只返回一个 0 到 1 之间的数字，不要返回其他内容。"

JUDGE_USER_PROMPT = """请评估以下股票分析 Agent 的回答质量，从 0 到 1 打分。

评分标准：
- 0.0-0.2: 回答为空、完全无关、或有明显错误
- 0.2-0.4: 提到了股票但信息不完整，缺少关键数据
- 0.4-0.6: 有基本信息（价格/涨跌），但分析粗糙
- 0.6-0.8: 信息完整（价格+涨跌+成交量等），分析有理据
- 0.8-1.0: 信息全面，推理清晰，包含技术面/基本面分析，有实用价值

额外扣分项：
- 工具调用了但信息没用到：-0.1
- 搞混了 A 股和港股：-0.2
- 编造了不存在的数据：-0.3

用户问题：{query}

Agent 回答：{answer}

请只返回一个 0 到 1 之间的小数（保留两位），不要解释。"""


class LLMJudge:
    """LLM-as-Judge 评分器"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.base_url = base_url
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def score(self, query: str, answer: str) -> float:
        """
        对 (query, answer) 打分

        Args:
            query: 用户查询
            answer: Agent 的最终回答

        Returns:
            0.0 ~ 1.0 的分数
        """
        if not answer or not answer.strip():
            return 0.0

        prompt = JUDGE_USER_PROMPT.format(query=query, answer=answer[:2000])

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            content = (resp.choices[0].message.content or "").strip()
            return self._parse_score(content)
        except Exception as exc:
            logger.warning(f"LLM Judge failed: {exc}")
            return 0.5  # 失败时返回中等分

    def _parse_score(self, text: str) -> float:
        """从模型输出中提取分数"""
        # 尝试直接 float
        try:
            score = float(text)
            return max(0.0, min(1.0, score))
        except ValueError:
            pass

        # 尝试正则提取数字
        match = re.search(r"(0\.\d+|1\.0|0|1)", text)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass

        logger.warning(f"Cannot parse score from: {text}")
        return 0.5

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """批量打分"""
        return [self.score(q, a) for q, a in pairs]
