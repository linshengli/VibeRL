from __future__ import annotations

import src.agentic.orchestrator as orchestrator_mod
from src.agentic.orchestrator import AgenticOrchestrator


def test_orchestrator_outputs_reasoning_and_reflection(monkeypatch) -> None:
    monkeypatch.setattr(
        orchestrator_mod,
        "search_stock_by_name",
        lambda name, market="all": {
            "results": [
                {"code": "600519", "name": "贵州茅台", "market": "a_share"},
            ]
        },
    )
    monkeypatch.setattr(
        orchestrator_mod,
        "get_realtime_quote",
        lambda code, market: {
            "code": code,
            "name": "贵州茅台",
            "price": 1688.0,
            "change_pct": 2.3,
            "volume": 123456,
        },
    )
    monkeypatch.setattr(
        orchestrator_mod,
        "get_technical_indicators",
        lambda code, market, indicators=None: {
            "code": code,
            "indicators": {
                "MA5": 1670.0,
                "MA20": 1622.0,
                "RSI14": 58.0,
                "MACD": {"histogram": 0.3},
            },
        },
    )
    monkeypatch.setattr(
        orchestrator_mod,
        "get_stock_info",
        lambda code, market: {
            "code": code,
            "name": "贵州茅台",
            "industry": "白酒",
            "market_cap_billion_cny": 22000.0,
            "listed_date": "2001-08-27",
        },
    )

    orchestrator = AgenticOrchestrator(model="rule-based", max_targets=1, max_steps_per_symbol=10)
    result = orchestrator.run("请分析茅台并给出推理和反思")

    assert result["num_targets"] == 1
    assert result["reflection"]
    analysis = result["analyses"][0]
    assert analysis["reasoning"]
    assert analysis["reflection"]
    assert "reasoning" in analysis["action_trace"]
    assert "reflection" in analysis["action_trace"]
    assert any(x["agent"] == "ReasoningAgent" for x in result["worklog"])
    assert any(x["agent"] == "ReflectAgent" for x in result["worklog"])

