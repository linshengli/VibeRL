from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from src.common.errors import MaxStepsExceeded, ToolExecutionError
from src.common.utils import try_json_loads
from src.models.entities import AgentMessage, AgentTrajectory, FunctionCall, ToolCall
from src.tools.stock_tools import get_stock_tool_registry, get_tool_definitions

DEFAULT_SYSTEM_PROMPT = (
    "你是一个股票分析助手。遵循 ReAct：先思考，再调用工具，再给出最终答案。"
    "若名称不明确，先调用 search_stock_by_name。"
)


class StockAnalysisAgent:
    def __init__(
        self,
        model: str = "rule-based",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.tools = get_stock_tool_registry()
        self.tool_defs = get_tool_definitions()

    def run(
        self,
        user_query: str,
        max_steps: int = 10,
        system_prompt: Optional[str] = None,
    ) -> AgentTrajectory:
        if not user_query.strip():
            raise ValueError("user_query cannot be empty")
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")

        messages: List[AgentMessage] = [
            AgentMessage(role="system", content=system_prompt or DEFAULT_SYSTEM_PROMPT),
            AgentMessage(role="user", content=user_query),
        ]
        tool_calls: List[ToolCall] = []

        for step in range(max_steps):
            assistant_content, planned_calls = self._next_action(messages)
            if planned_calls:
                messages.append(
                    AgentMessage(
                        role="assistant",
                        content=assistant_content,
                        tool_calls=planned_calls,
                    )
                )
                for call in planned_calls:
                    tool_calls.append(call)
                    tool_result = self._execute_tool_call(call)
                    messages.append(
                        AgentMessage(
                            role="tool",
                            name=call.function.name,
                            tool_call_id=call.id,
                            content=json.dumps(tool_result, ensure_ascii=False),
                        )
                    )
                continue

            final_output = (assistant_content or "").strip()
            if not final_output:
                final_output = "未能生成有效回答。"
            messages.append(AgentMessage(role="assistant", content=final_output))
            return AgentTrajectory(
                trajectory_id=str(uuid.uuid4()),
                prompt=user_query,
                messages=messages,
                tool_calls=tool_calls,
                final_output=final_output,
                num_turns=step + 1,
                success=True,
                metadata={"model": self.model},
            )

        raise MaxStepsExceeded(f"Agent exceeded max_steps={max_steps}")

    def _next_action(self, messages: List[AgentMessage]) -> Tuple[str, List[ToolCall]]:
        if self._can_use_openai():
            try:
                return self._next_action_openai(messages)
            except Exception:
                return self._next_action_rule_based(messages)
        return self._next_action_rule_based(messages)

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

    def _build_openai_messages(self, messages: List[AgentMessage]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for m in messages:
            payload: Dict[str, Any] = {"role": m.role}
            if m.content is not None:
                payload["content"] = m.content
            if m.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": c.type,
                        "function": {
                            "name": c.function.name,
                            "arguments": c.function.arguments,
                        },
                    }
                    for c in m.tool_calls
                ]
            if m.role == "tool":
                if m.tool_call_id:
                    payload["tool_call_id"] = m.tool_call_id
                if m.name:
                    payload["name"] = m.name
            result.append(payload)
        return result

    def _next_action_openai(self, messages: List[AgentMessage]) -> Tuple[str, List[ToolCall]]:
        from openai import OpenAI

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)

        resp = client.chat.completions.create(
            model=self.model,
            messages=self._build_openai_messages(messages),
            tools=self.tool_defs,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        content = msg.content or ""

        parsed_calls: List[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                parsed_calls.append(
                    ToolCall(
                        id=tc.id or f"call_{uuid.uuid4().hex[:8]}",
                        type=getattr(tc, "type", "function") or "function",
                        function=FunctionCall(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    )
                )
        return content, parsed_calls

    def _extract_targets(self, query: str) -> List[str]:
        query = query.lower()
        alias = {
            "茅台": "茅台",
            "贵州茅台": "贵州茅台",
            "腾讯": "腾讯",
            "平安": "平安",
            "阿里": "阿里",
            "阿里巴巴": "阿里巴巴",
        }
        targets: List[str] = []
        for key, val in alias.items():
            if key in query and val not in targets:
                targets.append(val)
        if not targets:
            targets.append("平安")
        return targets

    def _parse_tool_results(self, messages: List[AgentMessage], tool_name: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            if m.role == "tool" and m.name == tool_name and m.content:
                parsed = try_json_loads(m.content, {})
                if isinstance(parsed, dict):
                    out.append(parsed)
        return out

    def _already_called(self, messages: List[AgentMessage], tool_name: str) -> int:
        count = 0
        for m in messages:
            if m.tool_calls:
                for tc in m.tool_calls:
                    if tc.function.name == tool_name:
                        count += 1
        return count

    def _needs_indicator(self, query: str) -> bool:
        q = query.lower()
        keys = ["技术", "指标", "macd", "rsi", "kdj", "均线"]
        return any(k in q for k in keys)

    def _needs_stock_info(self, query: str) -> bool:
        q = query.lower()
        keys = ["基本", "行业", "市值", "公司", "上市"]
        return any(k in q for k in keys)

    def _next_action_rule_based(self, messages: List[AgentMessage]) -> Tuple[str, List[ToolCall]]:
        user_query = ""
        for m in messages:
            if m.role == "user" and m.content:
                user_query = m.content
                break

        targets = self._extract_targets(user_query)
        search_count = self._already_called(messages, "search_stock_by_name")
        quote_count = self._already_called(messages, "get_realtime_quote")
        info_count = self._already_called(messages, "get_stock_info")
        indicator_count = self._already_called(messages, "get_technical_indicators")

        if search_count < len(targets):
            name = targets[search_count]
            return (
                f"先搜索股票代码：{name}",
                [
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=FunctionCall(
                            name="search_stock_by_name",
                            arguments=json.dumps({"name": name, "market": "all"}, ensure_ascii=False),
                        ),
                    )
                ],
            )

        search_results = self._parse_tool_results(messages, "search_stock_by_name")
        selected: List[Dict[str, Any]] = []
        for item in search_results:
            rows = item.get("results", [])
            if rows:
                selected.append(rows[0])

        if quote_count < len(selected):
            stock = selected[quote_count]
            return (
                f"获取 {stock.get('name', stock.get('code'))} 的实时行情",
                [
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=FunctionCall(
                            name="get_realtime_quote",
                            arguments=json.dumps(
                                {
                                    "stock_code": stock["code"],
                                    "market": stock["market"],
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    )
                ],
            )

        if self._needs_stock_info(user_query) and info_count < len(selected):
            stock = selected[info_count]
            return (
                f"获取 {stock.get('name', stock.get('code'))} 的基础信息",
                [
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=FunctionCall(
                            name="get_stock_info",
                            arguments=json.dumps(
                                {
                                    "stock_code": stock["code"],
                                    "market": stock["market"],
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    )
                ],
            )

        if self._needs_indicator(user_query) and indicator_count < len(selected):
            stock = selected[indicator_count]
            return (
                f"获取 {stock.get('name', stock.get('code'))} 的技术指标",
                [
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=FunctionCall(
                            name="get_technical_indicators",
                            arguments=json.dumps(
                                {
                                    "stock_code": stock["code"],
                                    "market": stock["market"],
                                    "indicators": ["MA5", "MA20", "MACD", "RSI14"],
                                    "period": "daily",
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    )
                ],
            )

        return self._build_final_answer(messages), []

    def _build_final_answer(self, messages: List[AgentMessage]) -> str:
        quotes = self._parse_tool_results(messages, "get_realtime_quote")
        indicators = self._parse_tool_results(messages, "get_technical_indicators")
        infos = self._parse_tool_results(messages, "get_stock_info")

        if not quotes and not indicators and not infos:
            return "未获得有效工具结果，请提供更明确的股票名称或代码。"

        lines: List[str] = []
        for q in quotes:
            lines.append(
                f"{q.get('name', q.get('code'))}({q.get('code')}): 现价 {q.get('price')}, 涨跌幅 {q.get('change_pct')}%, 成交量 {q.get('volume')}"
            )

        for i in infos:
            lines.append(
                f"{i.get('name', i.get('code'))}: 行业 {i.get('industry')}, 市值约 {i.get('market_cap_billion_cny')} 亿元, 上市日期 {i.get('listed_date')}"
            )

        for ind in indicators:
            ind_map = ind.get("indicators", {})
            macd = ind_map.get("MACD", {}) if isinstance(ind_map.get("MACD"), dict) else {}
            lines.append(
                f"{ind.get('code')} 技术指标: MA5={ind_map.get('MA5')}, MA20={ind_map.get('MA20')}, RSI14={ind_map.get('RSI14')}, MACD(DIF={macd.get('dif')}, DEA={macd.get('dea')})"
            )

        return "\n".join(lines)

    def _execute_tool_call(self, call: ToolCall) -> Dict[str, Any]:
        tool_name = call.function.name
        tool = self.tools.get(tool_name)
        if tool is None:
            raise ToolExecutionError(f"Unknown tool: {tool_name}")

        args = try_json_loads(call.function.arguments, {})
        if not isinstance(args, dict):
            raise ToolExecutionError(f"Invalid arguments for tool {tool_name}")

        try:
            return tool.handler(**args)
        except TypeError as exc:
            raise ToolExecutionError(f"Invalid parameters for tool {tool_name}: {args}") from exc
        except Exception as exc:
            raise ToolExecutionError(f"Tool execution failed for {tool_name}: {exc}") from exc


__all__ = ["StockAnalysisAgent", "DEFAULT_SYSTEM_PROMPT", "AgentTrajectory", "AgentMessage", "ToolCall", "FunctionCall"]
