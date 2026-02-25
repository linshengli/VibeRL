from src.agent import StockAnalysisAgent
from src.reward import RewardComputer


def test_reward_computation() -> None:
    agent = StockAnalysisAgent(model="rule-based")
    trajectory = agent.run("查一下茅台的实时行情")

    rc = RewardComputer()
    reward = rc.compute(
        trajectory,
        expected_tools=["search_stock_by_name", "get_realtime_quote"],
    )

    assert 0.0 <= reward.final_reward <= 1.0
    assert reward.tool_correctness >= 0.0
    assert reward.output_quality >= 0.0
