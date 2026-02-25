from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from typing import Any, Dict, List, Sequence, Union

from src.agent import StockAnalysisAgent
from src.reward import RewardComputer


class AgentEvaluator:
    def evaluate(
        self,
        model_or_agent: Union[str, StockAnalysisAgent],
        test_cases: List[dict],
        metrics: Sequence[str] = ("tool_accuracy", "output_quality", "amb_accuracy"),
    ) -> Dict[str, Any]:
        agent = model_or_agent if isinstance(model_or_agent, StockAnalysisAgent) else StockAnalysisAgent(model=model_or_agent)
        reward_computer = RewardComputer()

        results: List[Dict[str, Any]] = []
        tool_hits: List[float] = []
        output_scores: List[float] = []
        amb_hits: List[float] = []

        for case in test_cases:
            query = case["query"]
            expected_tools = case.get("expected_tools", [])
            trajectory = agent.run(query)

            actual_tools = [tc.function.name for tc in trajectory.tool_calls]
            hit = 1.0 if self._tools_match(actual_tools, expected_tools) else 0.0
            tool_hits.append(hit)

            reward = reward_computer.compute(trajectory, expected_tools=expected_tools)
            output_scores.append(reward.output_quality / 0.4)

            is_amb = bool(case.get("is_ambiguous", False))
            if is_amb:
                amb_hits.append(hit)

            results.append(
                {
                    "query": query,
                    "expected_tools": expected_tools,
                    "actual_tools": actual_tools,
                    "tool_match": bool(hit),
                    "final_output": trajectory.final_output,
                    "reward": asdict(reward),
                }
            )

        summary: Dict[str, Any] = {
            "num_cases": len(test_cases),
            "num_ambiguous_cases": len(amb_hits),
            "metrics": {},
            "details": results,
        }

        if "tool_accuracy" in metrics:
            summary["metrics"]["tool_accuracy"] = round(mean(tool_hits) if tool_hits else 0.0, 4)
        if "output_quality" in metrics:
            summary["metrics"]["output_quality"] = round(mean(output_scores) if output_scores else 0.0, 4)
        if "amb_accuracy" in metrics:
            summary["metrics"]["amb_accuracy"] = round(mean(amb_hits) if amb_hits else 0.0, 4)

        return summary

    def _tools_match(self, actual: List[str], expected: List[str]) -> bool:
        if not expected:
            return True
        if len(actual) < len(expected):
            return False
        return actual[: len(expected)] == expected
