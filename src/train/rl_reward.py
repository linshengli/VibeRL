"""
RL Reward 计算模块 —— 给 verl GRPO 训练使用

职责：
1. 接收 verl rollout 产生的 (prompt, response) 对
2. 解析 response 中的 tool_calls
3. 执行工具拿到结果（或从 cache 中获取）
4. 计算 reward

verl 的 reward_fn 签名需要是:
    compute_reward(prompt: str, response: str) -> float

本模块还提供了更细粒度的 RewardConfig 和 AgentRewardComputer。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.reward.reward_computer import RewardComputer
from src.tools.stock_tools import get_stock_tool_registry

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """RL reward 超参数"""

    # 工具调用正确性权重 (0-0.4)
    tool_weight: float = 0.4
    # 输出质量权重 (0-0.4)
    output_weight: float = 0.4
    # 效率惩罚上限
    max_efficiency_penalty: float = 0.2
    # 超过多少轮开始惩罚
    efficiency_threshold: int = 3
    # 每超一轮的惩罚
    penalty_per_extra_turn: float = 0.02
    # 格式奖励（正确使用了 tool_calls 格式）
    format_bonus: float = 0.1
    # 是否使用 LLM-as-Judge（需要 API key）
    use_llm_judge: bool = False
    llm_judge_model: str = "deepseek-chat"
    llm_judge_weight: float = 0.2


class AgentRewardComputer:
    """
    Agent RL 场景的 reward 计算器

    与 src/reward/reward_computer.py 的区别：
    - reward_computer.py 操作 AgentTrajectory dataclass
    - 本类直接从 verl 的 raw text response 中解析 tool_calls
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.tools = get_stock_tool_registry()
        self._llm_judge = None

    def compute(
        self,
        prompt: str,
        response_text: str,
        expected_tools: Optional[List[str]] = None,
    ) -> float:
        """
        计算单个 (prompt, response) 的 reward

        Args:
            prompt: 用户查询
            response_text: 模型生成的完整回复（可能包含多轮 tool calling）
            expected_tools: 期望的工具调用序列（可选）

        Returns:
            float: 0.0 ~ 1.0 的 reward
        """
        # 解析 response 中的结构
        tool_calls, final_output, num_turns = self._parse_response(response_text)

        reward = 0.0

        # 1. 工具调用正确性
        tool_score = self._tool_score(tool_calls, expected_tools)
        reward += tool_score

        # 2. 输出质量
        output_score = self._output_score(final_output, prompt)
        reward += output_score

        # 3. 效率惩罚
        efficiency = self._efficiency_penalty(num_turns)
        reward += efficiency

        # 4. 格式奖励
        if tool_calls:
            reward += self.config.format_bonus

        # 5. LLM-as-Judge（可选）
        if self.config.use_llm_judge and final_output:
            judge_score = self._llm_judge_score(prompt, final_output)
            reward += judge_score * self.config.llm_judge_weight

        return max(0.0, min(1.0, reward))

    def _parse_response(self, response_text: str) -> Tuple[List[Dict], str, int]:
        """
        从模型的 raw response 中解析 tool_calls 和 final_output

        verl 的 rollout 可能产生的格式：
        1. 纯文本回答（没有工具调用）
        2. 带 <tool_call>...</tool_call> 标记的文本
        3. JSON 格式的 function calling
        """
        tool_calls: List[Dict] = []
        final_output = response_text
        num_turns = 1

        # 尝试解析 JSON 格式的 tool_calls
        # 模型可能输出类似:
        # {"name": "search_stock_by_name", "arguments": {"name": "茅台"}}
        tool_call_pattern = r'\{"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\}'
        matches = re.findall(tool_call_pattern, response_text)
        for name, args_str in matches:
            try:
                args = json.loads(args_str)
                tool_calls.append({"name": name, "arguments": args})
            except json.JSONDecodeError:
                pass

        # 也尝试解析 <tool_call> 标记
        tc_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        tc_matches = re.findall(tc_pattern, response_text, re.DOTALL)
        for tc_text in tc_matches:
            try:
                tc_data = json.loads(tc_text)
                if isinstance(tc_data, dict) and "name" in tc_data:
                    tool_calls.append(tc_data)
            except json.JSONDecodeError:
                pass

        # 提取最终输出（最后一段非工具调用的文本）
        # 去掉工具调用相关的部分
        cleaned = re.sub(tool_call_pattern, "", response_text)
        cleaned = re.sub(tc_pattern, "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()
        if cleaned:
            final_output = cleaned

        num_turns = max(1, len(tool_calls) + 1)

        return tool_calls, final_output, num_turns

    def _tool_score(self, actual_calls: List[Dict], expected: Optional[List[str]]) -> float:
        """工具调用正确性评分"""
        actual_names = [c.get("name", "") for c in actual_calls]

        if expected is None or len(expected) == 0:
            # 没有 expected_tools，只要调用了工具就给分
            if actual_names:
                return self.config.tool_weight * 0.8
            return 0.0

        if not actual_names:
            return 0.0

        # 计算匹配度
        matched = 0
        for i, name in enumerate(actual_names):
            if i < len(expected) and name == expected[i]:
                matched += 1

        ratio = matched / len(expected)
        return self.config.tool_weight * ratio

    def _output_score(self, final_output: str, prompt: str) -> float:
        """输出质量评分"""
        text = (final_output or "").strip()
        if not text or len(text) < 10:
            return 0.0

        score = 0.0
        max_score = self.config.output_weight

        # 长度合理
        if len(text) >= 30:
            score += max_score * 0.2
        if len(text) >= 100:
            score += max_score * 0.1

        # 包含数字（股价、指标等）
        if any(ch.isdigit() for ch in text):
            score += max_score * 0.25

        # 包含关键词
        keywords = ["价格", "涨跌", "行情", "指标", "成交", "市值", "建议"]
        if any(kw in text for kw in keywords):
            score += max_score * 0.25

        # 结构化（有换行或括号）
        if "\n" in text or ("(" in text and ")" in text):
            score += max_score * 0.2

        return min(max_score, score)

    def _efficiency_penalty(self, num_turns: int) -> float:
        """效率惩罚"""
        if num_turns <= self.config.efficiency_threshold:
            return 0.0
        extra = num_turns - self.config.efficiency_threshold
        penalty = min(self.config.max_efficiency_penalty, extra * self.config.penalty_per_extra_turn)
        return -penalty

    def _llm_judge_score(self, prompt: str, answer: str) -> float:
        """使用 LLM 评分（可选）"""
        if self._llm_judge is None:
            try:
                from src.reward.llm_judge import LLMJudge
                api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
                if api_key:
                    self._llm_judge = LLMJudge(model=self.config.llm_judge_model, api_key=api_key)
            except Exception:
                pass

        if self._llm_judge is None:
            return 0.5  # 默认中等分

        try:
            return self._llm_judge.score(prompt, answer)
        except Exception:
            return 0.5


# =============================================================
#  verl 接口函数
#  verl 的 reward_fn 配置指向这个函数
# =============================================================

_reward_computer: Optional[AgentRewardComputer] = None


def compute_reward_for_verl(prompt: str, response: str) -> float:
    """
    verl GRPO 训练调用的 reward 函数

    签名: (prompt: str, response: str) -> float
    """
    global _reward_computer
    if _reward_computer is None:
        _reward_computer = AgentRewardComputer()

    return _reward_computer.compute(prompt, response)
