from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import StockAnalysisAgent
from src.common.utils import try_json_loads
from src.models.entities import AgentTrajectory
from src.tools.stock_tools import get_stock_info, search_stock_by_name


@dataclass
class TargetTask:
    name: str
    need_quote: bool = True
    need_indicator: bool = True
    need_info: bool = True


class PlannerAgent:
    def plan(self, user_query: str, max_targets: int = 3) -> List[TargetTask]:
        aliases = [
            "茅台",
            "贵州茅台",
            "腾讯",
            "平安",
            "平安银行",
            "中国平安",
            "阿里",
            "阿里巴巴",
        ]

        selected: List[str] = []
        for name in aliases:
            if name in user_query and name not in selected:
                selected.append(name)

        if not selected:
            selected = ["茅台"]

        selected = selected[:max_targets]
        need_indicator = any(k in user_query for k in ["技术", "指标", "MACD", "RSI", "KDJ", "均线"])

        return [
            TargetTask(
                name=s,
                need_quote=True,
                need_indicator=need_indicator,
                need_info=True,
            )
            for s in selected
        ]


class ResearchAgent:
    def __init__(self, model: str, debug_proxy: Optional[str] = None) -> None:
        self.agent = StockAnalysisAgent(model=model, base_url=debug_proxy)

    def run(self, task: TargetTask) -> Dict[str, Any]:
        query = f"请分析{task.name}的实时行情"
        if task.need_indicator:
            query += "和关键技术指标(MA5,MA20,MACD,RSI14)"
        query += "，并给出简短结论。"

        trajectory = self.agent.run(query)
        parsed = self._parse_trajectory(trajectory)
        parsed["query"] = query
        parsed["final_answer"] = trajectory.final_output
        parsed["trajectory_id"] = trajectory.trajectory_id
        return parsed

    def _parse_trajectory(self, trajectory: AgentTrajectory) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "search": None,
            "quote": None,
            "indicators": None,
        }

        for msg in trajectory.messages:
            if msg.role != "tool" or not msg.name or not msg.content:
                continue
            payload = try_json_loads(msg.content, {})
            if not isinstance(payload, dict):
                continue

            if msg.name == "search_stock_by_name" and out["search"] is None:
                results = payload.get("results", [])
                if results:
                    out["search"] = results[0]
            elif msg.name == "get_realtime_quote":
                out["quote"] = payload
            elif msg.name == "get_technical_indicators":
                out["indicators"] = payload

        return out


class FundamentalAgent:
    def run(self, search_hit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not search_hit:
            return {"error": "no_symbol", "message": "未找到代码，无法查询基础信息"}

        code = str(search_hit.get("code", ""))
        market = str(search_hit.get("market", ""))
        if not code or market not in {"a_share", "hk_share"}:
            return {"error": "invalid_symbol", "message": "代码或市场不完整"}

        return get_stock_info(stock_code=code, market=market)


class RiskAgent:
    def run(self, quote: Optional[Dict[str, Any]], indicators: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not quote:
            return {
                "risk_level": "unknown",
                "risk_score": None,
                "reasons": ["缺少实时行情，无法评估"],
                "suggestion": "先补齐行情数据",
            }

        reasons: List[str] = []
        score = 0

        change_pct = quote.get("change_pct")
        if isinstance(change_pct, (int, float)):
            abs_change = abs(float(change_pct))
            if abs_change >= 5:
                score += 2
                reasons.append(f"当日波动较大({change_pct}%)")
            elif abs_change >= 2:
                score += 1
                reasons.append(f"当日波动中等({change_pct}%)")

        ind_map = indicators.get("indicators", {}) if isinstance(indicators, dict) else {}
        rsi = ind_map.get("RSI14")
        if isinstance(rsi, (int, float)):
            if rsi >= 70:
                score += 1
                reasons.append(f"RSI14 偏高({rsi})，有短期过热风险")
            elif rsi <= 30:
                score += 1
                reasons.append(f"RSI14 偏低({rsi})，波动可能加剧")

        macd = ind_map.get("MACD") if isinstance(ind_map.get("MACD"), dict) else {}
        hist = macd.get("histogram")
        if isinstance(hist, (int, float)) and float(hist) < 0:
            score += 1
            reasons.append("MACD 柱体为负，短线动能偏弱")

        if score >= 3:
            level = "high"
            suggestion = "建议降低仓位并设置止损，等待趋势确认"
        elif score >= 1:
            level = "medium"
            suggestion = "建议控制仓位，观察量价与指标共振信号"
        else:
            level = "low"
            suggestion = "风险相对可控，关注后续基本面与成交量变化"

        if not reasons:
            reasons.append("短线波动和技术指标未见明显异常")

        return {
            "risk_level": level,
            "risk_score": score,
            "reasons": reasons,
            "suggestion": suggestion,
        }


class ReporterAgent:
    @staticmethod
    def _as_dict(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def render(self, user_query: str, analyses: List[Dict[str, Any]]) -> str:
        lines = [
            f"用户需求: {user_query}",
            "",
            "多 Agent 协作结论:",
        ]

        for idx, item in enumerate(analyses, start=1):
            symbol = self._as_dict(item.get("symbol"))
            quote = self._as_dict(item.get("quote"))
            info = self._as_dict(item.get("info"))
            risk = self._as_dict(item.get("risk"))

            name = symbol.get("name") or quote.get("name") or info.get("name") or f"标的{idx}"
            code = symbol.get("code") or quote.get("code") or info.get("code") or "-"

            lines.append(f"{idx}. {name} ({code})")
            if quote:
                lines.append(
                    f"   行情: 现价 {quote.get('price')}, 涨跌幅 {quote.get('change_pct')}%, 量 {quote.get('volume')}"
                )
            if info and not info.get("error"):
                lines.append(
                    f"   基本面: 行业 {info.get('industry')}, 市值(亿元) {info.get('market_cap_billion_cny')}, 上市 {info.get('listed_date')}"
                )

            indicator_container = self._as_dict(item.get("indicators"))
            indicators = self._as_dict(indicator_container.get("indicators"))
            if indicators:
                lines.append(
                    f"   技术: MA5={indicators.get('MA5')}, MA20={indicators.get('MA20')}, RSI14={indicators.get('RSI14')}, MACD={indicators.get('MACD')}"
                )

            lines.append(
                f"   风险: {risk.get('risk_level')} (score={risk.get('risk_score')}) | 建议: {risk.get('suggestion')}"
            )

        lines.append("")
        lines.append("执行说明: Planner -> Research -> Fundamental -> Risk -> Reporter")
        return "\n".join(lines)


class MultiAgentOrchestrator:
    def __init__(self, model: str, debug_proxy: Optional[str] = None, max_targets: int = 3) -> None:
        self.max_targets = max_targets
        self.planner = PlannerAgent()
        self.researcher = ResearchAgent(model=model, debug_proxy=debug_proxy)
        self.fundamental = FundamentalAgent()
        self.risk = RiskAgent()
        self.reporter = ReporterAgent()

    def run(self, user_query: str) -> Dict[str, Any]:
        tasks = self.planner.plan(user_query, max_targets=self.max_targets)
        worklog: List[Dict[str, Any]] = [
            {
                "agent": "PlannerAgent",
                "title": "拆解任务",
                "detail": {"tasks": [task.__dict__ for task in tasks]},
            }
        ]
        analyses: List[Dict[str, Any]] = []

        for task in tasks:
            research_result = self.researcher.run(task)
            symbol = research_result.get("search")
            worklog.append(
                {
                    "agent": "ResearchAgent",
                    "title": f"调研 {task.name}",
                    "detail": {
                        "query": research_result.get("query"),
                        "trajectory_id": research_result.get("trajectory_id"),
                        "symbol": symbol,
                    },
                }
            )

            # 若 research 未成功拿到代码，额外再走一次名称检索兜底
            if not symbol:
                fallback = search_stock_by_name(task.name)
                rows = fallback.get("results", []) if isinstance(fallback, dict) else []
                if rows:
                    symbol = rows[0]
                    worklog.append(
                        {
                            "agent": "ResearchAgent",
                            "title": f"{task.name} 兜底检索",
                            "detail": {"symbol": symbol},
                        }
                    )

            info = self.fundamental.run(symbol)
            worklog.append(
                {
                    "agent": "FundamentalAgent",
                    "title": f"{task.name} 基本面",
                    "detail": info,
                }
            )
            risk = self.risk.run(research_result.get("quote"), research_result.get("indicators"))
            worklog.append(
                {
                    "agent": "RiskAgent",
                    "title": f"{task.name} 风险评估",
                    "detail": risk,
                }
            )

            analyses.append(
                {
                    "task": task.__dict__,
                    "symbol": symbol,
                    "quote": research_result.get("quote"),
                    "indicators": research_result.get("indicators"),
                    "info": info,
                    "risk": risk,
                    "final_answer": research_result.get("final_answer"),
                    "trajectory_id": research_result.get("trajectory_id"),
                }
            )

        report = self.reporter.render(user_query, analyses)
        worklog.append(
            {
                "agent": "ReporterAgent",
                "title": "生成总结",
                "detail": {"report_preview": report[:300]},
            }
        )
        return {
            "query": user_query,
            "num_targets": len(tasks),
            "analyses": analyses,
            "report": report,
            "worklog": worklog,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent stock analysis demo")
    parser.add_argument("--query", required=True, help="用户查询")
    parser.add_argument("--model", default="rule-based", help="模型名，如 deepseek-chat 或 rule-based")
    parser.add_argument("--debug-proxy", default=None, help="可选 OpenAI 兼容代理地址")
    parser.add_argument("--max-targets", type=int, default=3, help="最多分析的股票数量")
    parser.add_argument("--json-output", default=None, help="可选 JSON 输出路径")
    args = parser.parse_args()

    orchestrator = MultiAgentOrchestrator(
        model=args.model,
        debug_proxy=args.debug_proxy,
        max_targets=args.max_targets,
    )
    result = orchestrator.run(args.query)

    print(result["report"])

    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON 已写入: {path}")


if __name__ == "__main__":
    main()
