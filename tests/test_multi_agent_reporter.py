from __future__ import annotations

from src.demo.multi_agent_demo import ReporterAgent


def test_reporter_handles_none_fields() -> None:
    reporter = ReporterAgent()
    text = reporter.render(
        "比较茅台",
        [
            {
                "symbol": None,
                "quote": {"name": "贵州茅台", "code": "600519", "price": 1234.5, "change_pct": 1.2, "volume": 1000},
                "indicators": None,
                "info": None,
                "risk": {"risk_level": "low", "risk_score": 0, "suggestion": "观察"},
            }
        ],
    )

    assert "贵州茅台 (600519)" in text
    assert "风险: low" in text
