from __future__ import annotations

from typing import List, Optional

from src.models.entities import AgentTrajectory, RewardSignal


class RewardComputer:
    def compute(
        self,
        trajectory: AgentTrajectory,
        expected_tools: Optional[List[str]] = None,
        use_llm_judge: bool = False,
    ) -> RewardSignal:
        _ = use_llm_judge

        actual_tools = [call.function.name for call in trajectory.tool_calls]
        expected = expected_tools or trajectory.metadata.get("expected_tools", []) or []

        tool_correctness, detail = self._tool_score(actual_tools, expected)
        output_quality = self._output_score(trajectory.final_output)
        efficiency_penalty = self._efficiency_penalty(trajectory.num_turns)

        final_reward = max(0.0, min(1.0, tool_correctness + output_quality + efficiency_penalty))
        return RewardSignal(
            trajectory_id=trajectory.trajectory_id,
            tool_correctness=tool_correctness,
            output_quality=output_quality,
            efficiency_penalty=efficiency_penalty,
            final_reward=final_reward,
            expected_tools=expected,
            actual_tools=actual_tools,
            tool_match_detail=detail,
        )

    def _tool_score(self, actual: List[str], expected: List[str]) -> tuple[float, List[bool]]:
        if not expected:
            return (0.4 if actual else 0.0), [True for _ in actual]

        matched = 0
        detail: List[bool] = []
        for idx, tool in enumerate(actual):
            ok = idx < len(expected) and expected[idx] == tool
            detail.append(ok)
            if ok:
                matched += 1

        ratio = matched / len(expected)
        return min(0.4, max(0.0, ratio * 0.4)), detail

    def _output_score(self, final_output: str) -> float:
        text = (final_output or "").strip()
        if not text:
            return 0.0

        score = 0.0
        if len(text) >= 20:
            score += 0.1
        if any(ch.isdigit() for ch in text):
            score += 0.1
        if "涨跌" in text or "价格" in text or "指标" in text:
            score += 0.1
        if "(" in text and ")" in text:
            score += 0.1
        return min(0.4, score)

    def _efficiency_penalty(self, num_turns: int) -> float:
        if num_turns <= 3:
            return 0.0
        penalty = min(0.2, (num_turns - 3) * 0.02)
        return -penalty
