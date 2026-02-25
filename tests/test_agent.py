from src.agent import StockAnalysisAgent


def test_agent_runs_with_rule_based_strategy() -> None:
    agent = StockAnalysisAgent(model="rule-based")
    trajectory = agent.run("帮我查一下茅台最近的技术指标")

    assert trajectory.success is True
    called_tools = [call.function.name for call in trajectory.tool_calls]
    assert "search_stock_by_name" in called_tools
    assert "get_realtime_quote" in called_tools
    assert "get_technical_indicators" in called_tools
    assert trajectory.final_output
