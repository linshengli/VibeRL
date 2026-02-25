from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.tools.stock_tools import (
    get_realtime_quote,
    get_stock_info,
    get_technical_indicators,
    search_stock_by_name,
)


@dataclass
class SymbolState:
    target_name: str
    symbol: Optional[Dict[str, Any]] = None
    quote: Optional[Dict[str, Any]] = None
    indicators: Optional[Dict[str, Any]] = None
    info: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    risk: Optional[Dict[str, Any]] = None
    reflection: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    action_trace: List[str] = field(default_factory=list)
    steps: int = 0


@dataclass
class Plan:
    targets: List[str]
    need_quote: bool
    need_indicator: bool
    need_info: bool
    need_reasoning: bool
    need_risk: bool
    need_reflection: bool


class AgenticOrchestrator:
    """A dynamic, state-driven multi-agent orchestrator.

    It does not run a rigid fixed pipeline. Instead, it chooses next action
    based on current state + user goals at each step.
    """

    def __init__(
        self,
        max_targets: int = 3,
        max_steps_per_symbol: int = 8,
        model: str = "rule-based",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.max_targets = max_targets
        self.max_steps_per_symbol = max_steps_per_symbol
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url

    def run(
        self,
        user_query: str,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        def emit(event: Dict[str, Any]) -> None:
            if event_callback is not None:
                event_callback(event)

        plan = self._build_plan(user_query)
        emit({"type": "planner", "title": "任务规划", "detail": {**plan.__dict__, "mode": self.model}})

        worklog: List[Dict[str, Any]] = [
            {
                "agent": "PlannerAgent",
                "title": "任务规划",
                "detail": {**plan.__dict__, "mode": self.model},
            }
        ]
        analyses: List[Dict[str, Any]] = []

        for target in plan.targets:
            state = SymbolState(target_name=target)
            emit({"type": "symbol.start", "symbol": target})

            while state.steps < self.max_steps_per_symbol:
                state.steps += 1
                action = self._next_action(plan, state)
                if action is None:
                    break

                result = self._execute_action(action, state, plan)
                state.action_trace.append(action)
                worklog.append(result["log"])
                emit(
                    {
                        "type": "agent.step",
                        "symbol": target,
                        "action": action,
                        "detail": result["log"],
                    }
                )

                if result.get("error"):
                    state.notes.append(str(result["error"]))

            analyses.append(
                {
                    "task": {
                        "name": target,
                        "need_quote": plan.need_quote,
                        "need_indicator": plan.need_indicator,
                        "need_info": plan.need_info,
                        "need_reasoning": plan.need_reasoning,
                        "need_reflection": plan.need_reflection,
                    },
                    "symbol": state.symbol,
                    "quote": state.quote,
                    "indicators": state.indicators,
                    "info": state.info,
                    "reasoning": state.reasoning,
                    "risk": state.risk,
                    "reflection": state.reflection,
                    "action_trace": state.action_trace,
                    "notes": state.notes,
                }
            )

        reflection = self._aggregate_reflection(analyses)
        worklog.append(
            {
                "agent": "ReflectAgent",
                "title": "全局反思",
                "detail": reflection,
            }
        )
        emit({"type": "reflection.done", "detail": reflection})

        report = self._render_report(user_query, analyses, reflection)
        worklog.append(
            {
                "agent": "ReporterAgent",
                "title": "汇总报告",
                "detail": {"report_preview": report[:300]},
            }
        )
        emit({"type": "report.done", "report_preview": report[:120]})

        return {
            "query": user_query,
            "num_targets": len(plan.targets),
            "analyses": analyses,
            "reflection": reflection,
            "report": report,
            "worklog": worklog,
        }

    def _build_plan(self, user_query: str) -> Plan:
        aliases = ["茅台", "贵州茅台", "腾讯", "平安", "平安银行", "中国平安", "阿里", "阿里巴巴"]
        targets: List[str] = []
        for name in aliases:
            if name in user_query and name not in targets:
                targets.append(name)
        if not targets:
            targets = ["茅台"]

        need_indicator = any(k in user_query.lower() for k in ["技术", "指标", "macd", "rsi", "kdj", "均线"])
        need_info = any(k in user_query for k in ["基本面", "行业", "公司", "市值", "上市", "介绍"]) or True
        need_risk = any(k in user_query.lower() for k in ["风险", "建议", "策略"]) or True

        return Plan(
            targets=targets[: self.max_targets],
            need_quote=True,
            need_indicator=need_indicator,
            need_info=need_info,
            need_reasoning=True,
            need_risk=need_risk,
            need_reflection=True,
        )

    def _next_action(self, plan: Plan, state: SymbolState) -> Optional[str]:
        candidates: List[str] = []

        if state.symbol is None:
            candidates.append("search")
        if plan.need_quote and state.symbol is not None and state.quote is None:
            candidates.append("quote")
        if plan.need_indicator and state.symbol is not None and state.indicators is None:
            candidates.append("indicators")
        if plan.need_info and state.symbol is not None and state.info is None:
            candidates.append("info")
        if (
            plan.need_reasoning
            and state.symbol is not None
            and state.reasoning is None
            and (state.quote is not None or state.indicators is not None or state.info is not None)
        ):
            candidates.append("reasoning")
        if plan.need_risk and state.risk is None and (state.quote is not None or state.indicators is not None):
            candidates.append("risk")
        if plan.need_reflection and state.reflection is None and (state.reasoning is not None or state.risk is not None):
            candidates.append("reflection")

        if not candidates:
            return None

        # Agentic decision: prioritize missing high-value signals but adapt to current state.
        priority = ["search", "quote", "indicators", "info", "reasoning", "risk", "reflection"]
        for item in priority:
            if item in candidates:
                return item
        return candidates[0]

    def _execute_action(self, action: str, state: SymbolState, plan: Plan) -> Dict[str, Any]:
        if action == "search":
            res = search_stock_by_name(state.target_name, market="all")
            rows = res.get("results", []) if isinstance(res, dict) else []
            if rows:
                state.symbol = rows[0]
            return {
                "log": {"agent": "SearchAgent", "title": f"搜索 {state.target_name}", "detail": res},
                "error": res if isinstance(res, dict) and res.get("error") else None,
            }

        if state.symbol is None:
            return {
                "log": {"agent": "System", "title": "缺少 symbol", "detail": {"action": action}},
                "error": "symbol_missing",
            }

        code = str(state.symbol.get("code", ""))
        market = str(state.symbol.get("market", ""))

        if action == "quote":
            res = get_realtime_quote(code, market)
            if not res.get("error"):
                state.quote = res
            return {
                "log": {"agent": "MarketAgent", "title": f"行情 {code}", "detail": res},
                "error": res if res.get("error") else None,
            }

        if action == "indicators":
            res = get_technical_indicators(code, market, indicators=["MA5", "MA20", "MACD", "RSI14"])
            if not res.get("error"):
                state.indicators = res
            return {
                "log": {"agent": "TechnicalAgent", "title": f"指标 {code}", "detail": res},
                "error": res if res.get("error") else None,
            }

        if action == "info":
            res = get_stock_info(code, market)
            if not res.get("error"):
                state.info = res
            return {
                "log": {"agent": "FundamentalAgent", "title": f"基本面 {code}", "detail": res},
                "error": res if res.get("error") else None,
            }

        if action == "reasoning":
            reasoning = self._reason_for_symbol(state, plan)
            state.reasoning = reasoning
            return {
                "log": {"agent": "ReasoningAgent", "title": f"推理 {code}", "detail": {"reasoning": reasoning}},
                "error": None,
            }

        if action == "risk":
            risk = self._compute_risk(state.quote, state.indicators)
            state.risk = risk
            return {
                "log": {"agent": "RiskAgent", "title": f"风险评估 {code}", "detail": risk},
                "error": None,
            }

        if action == "reflection":
            reflection = self._reflect_symbol(state)
            state.reflection = reflection
            return {
                "log": {"agent": "ReflectAgent", "title": f"反思 {code}", "detail": {"reflection": reflection}},
                "error": None,
            }

        return {
            "log": {"agent": "System", "title": "未知动作", "detail": {"action": action}},
            "error": "unknown_action",
        }

    def _can_use_openai(self) -> bool:
        if self.model in {"rule-based", "mock", "offline"}:
            return False
        if not self.api_key:
            return False
        try:
            import openai  # noqa: F401
        except Exception:
            return False
        return True

    def _llm_complete(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self._can_use_openai():
            return None

        try:
            from openai import OpenAI

            client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            client = OpenAI(**client_kwargs)
            resp = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content
            if not content:
                return None
            return str(content).strip()
        except Exception:
            return None

    def _reason_for_symbol(self, state: SymbolState, plan: Plan) -> str:
        snapshot = {
            "target_name": state.target_name,
            "symbol": state.symbol,
            "quote": state.quote,
            "indicators": state.indicators,
            "info": state.info,
            "need_indicator": plan.need_indicator,
            "need_info": plan.need_info,
        }
        prompt = (
            "请根据以下股票信息给出简洁推理，格式为两行：\n"
            "1) 关键证据: ...\n"
            "2) 推理结论: ...\n\n"
            f"数据: {json.dumps(snapshot, ensure_ascii=False)}"
        )
        llm_text = self._llm_complete(
            system_prompt="你是股票分析推理助手。禁止编造数据，缺失就明确说明。",
            user_prompt=prompt,
        )
        if llm_text:
            return llm_text

        evidence: List[str] = []
        quote = state.quote or {}
        if isinstance(quote.get("price"), (int, float)):
            evidence.append(f"现价 {quote.get('price')}")
        if isinstance(quote.get("change_pct"), (int, float)):
            evidence.append(f"涨跌幅 {quote.get('change_pct')}%")

        ind_map = state.indicators.get("indicators", {}) if isinstance(state.indicators, dict) else {}
        if isinstance(ind_map.get("RSI14"), (int, float)):
            evidence.append(f"RSI14={ind_map.get('RSI14')}")
        macd = ind_map.get("MACD") if isinstance(ind_map.get("MACD"), dict) else {}
        if isinstance(macd.get("histogram"), (int, float)):
            evidence.append(f"MACD柱体={macd.get('histogram')}")

        if isinstance(state.info, dict) and not state.info.get("error"):
            industry = state.info.get("industry")
            if industry:
                evidence.append(f"行业={industry}")
            mcap = state.info.get("market_cap_billion_cny")
            if isinstance(mcap, (int, float)):
                evidence.append(f"市值(亿元)={mcap}")

        if not evidence:
            evidence.append("暂无足够结构化数据")

        conclusion = "建议继续观察并结合风险评估结果再决策"
        rsi = ind_map.get("RSI14")
        if isinstance(rsi, (int, float)):
            if rsi >= 70:
                conclusion = "短线可能偏热，避免追高，等待回撤确认"
            elif rsi <= 30:
                conclusion = "短线处于偏弱区域，关注是否出现反转信号"

        change_pct = quote.get("change_pct")
        if isinstance(change_pct, (int, float)) and abs(float(change_pct)) >= 5:
            conclusion = "当日波动显著，应控制仓位并设置止损"

        return f"关键证据: {'; '.join(str(x) for x in evidence)}\n推理结论: {conclusion}"

    def _reflect_symbol(self, state: SymbolState) -> str:
        snapshot = {
            "target_name": state.target_name,
            "symbol": state.symbol,
            "reasoning": state.reasoning,
            "risk": state.risk,
            "notes": state.notes,
            "action_trace": state.action_trace,
        }
        prompt = (
            "请对该标的分析做一次反思，总结三点：\n"
            "1) 结论可靠性\n"
            "2) 数据缺口\n"
            "3) 下一步动作\n\n"
            f"数据: {json.dumps(snapshot, ensure_ascii=False)}"
        )
        llm_text = self._llm_complete(
            system_prompt="你是审慎的分析复盘助手，强调证据和不确定性。",
            user_prompt=prompt,
        )
        if llm_text:
            return llm_text

        risk_level = ((state.risk or {}).get("risk_level") if isinstance(state.risk, dict) else None) or "unknown"
        missing: List[str] = []
        if state.quote is None:
            missing.append("缺少行情")
        if state.info is None:
            missing.append("缺少基本面")
        if state.indicators is None:
            missing.append("缺少技术指标")
        if not missing:
            missing.append("关键字段基本齐全")

        reliability = "中等"
        if risk_level == "unknown" or state.reasoning is None:
            reliability = "偏低"
        elif risk_level == "low" and not state.notes:
            reliability = "中高"

        next_step = "补齐缺失数据后再评估"
        if missing == ["关键字段基本齐全"]:
            next_step = "结合交易计划设置入场/止损规则并持续跟踪"

        return (
            f"可靠性: {reliability}\n"
            f"数据缺口: {', '.join(missing)}\n"
            f"下一步: {next_step}"
        )

    def _aggregate_reflection(self, analyses: List[Dict[str, Any]]) -> str:
        if not analyses:
            return "无可复盘标的。"

        lines: List[str] = []
        for idx, item in enumerate(analyses, start=1):
            symbol = item.get("symbol") if isinstance(item.get("symbol"), dict) else {}
            name = symbol.get("name") or symbol.get("code") or f"标的{idx}"
            reflection = str(item.get("reflection") or "").strip() or "暂无反思内容"
            lines.append(f"{idx}. {name}: {reflection}")
        lines.append("总体建议: 先看数据完整性，再看风险级别，最后才做仓位决策。")
        return "\n".join(lines)

    def _compute_risk(self, quote: Optional[Dict[str, Any]], indicators: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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

    def _render_report(self, user_query: str, analyses: List[Dict[str, Any]], reflection: str) -> str:
        lines = [f"用户需求: {user_query}", "", "Agentic 协作结论:"]

        for idx, item in enumerate(analyses, start=1):
            symbol = item.get("symbol") if isinstance(item.get("symbol"), dict) else {}
            quote = item.get("quote") if isinstance(item.get("quote"), dict) else {}
            info = item.get("info") if isinstance(item.get("info"), dict) else {}
            risk = item.get("risk") if isinstance(item.get("risk"), dict) else {}
            indicators = (
                item.get("indicators", {}).get("indicators", {})
                if isinstance(item.get("indicators"), dict)
                else {}
            )

            name = symbol.get("name") or quote.get("name") or info.get("name") or f"标的{idx}"
            code = symbol.get("code") or quote.get("code") or info.get("code") or "-"

            lines.append(f"{idx}. {name} ({code})")
            if quote:
                lines.append(f"   行情: 现价 {quote.get('price')}, 涨跌幅 {quote.get('change_pct')}%, 量 {quote.get('volume')}")
            if info and not info.get("error"):
                lines.append(f"   基本面: 行业 {info.get('industry')}, 市值(亿元) {info.get('market_cap_billion_cny')}, 上市 {info.get('listed_date')}")
            if indicators:
                lines.append(
                    f"   技术: MA5={indicators.get('MA5')}, MA20={indicators.get('MA20')}, RSI14={indicators.get('RSI14')}, MACD={indicators.get('MACD')}"
                )
            reasoning = str(item.get("reasoning") or "").strip()
            if reasoning:
                lines.append(f"   推理: {reasoning}")
            if risk:
                lines.append(f"   风险: {risk.get('risk_level')} (score={risk.get('risk_score')}) | 建议: {risk.get('suggestion')}")
            reflection_item = str(item.get("reflection") or "").strip()
            if reflection_item:
                lines.append(f"   反思: {reflection_item}")

            notes = item.get("notes") or []
            if notes:
                lines.append(f"   备注: {'; '.join(str(x) for x in notes)}")

        lines.append("")
        lines.append("全局反思:")
        lines.append(reflection)
        lines.append("")
        lines.append("执行说明: 动态决策循环（支持 reasoning / reflection，非固定流程）")
        return "\n".join(lines)
