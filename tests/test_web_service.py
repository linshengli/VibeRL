from __future__ import annotations

import json

from src.models.entities import AgentMessage, AgentTrajectory, FunctionCall, ToolCall
from src.web.service import AgentQueryService


def test_service_degrades_when_multi_agent_fails() -> None:
    def chat_runner(query: str, model: str):
        return {
            "trajectory_id": "t1",
            "final_answer": f"ok:{query}",
            "num_turns": 1,
            "success": True,
            "tool_calls": [],
            "messages": [],
            "tool_trace": [],
        }

    def bad_multi(query: str, model: str):
        raise RuntimeError("multi failed")

    svc = AgentQueryService(default_model="rule-based", chat_runner=chat_runner, multi_runner=bad_multi)
    result = svc.execute("hello")

    assert result["status"] == "ok"
    assert result["chat"]["final_answer"] == "ok:hello"
    assert result["multi_agent"]["error"]["code"] == "MULTI_AGENT_FAILED"
    assert result["warnings"][0]["code"] == "MULTI_AGENT_FAILED"


def test_service_uses_conversation_history_context() -> None:
    captured = {"query": ""}

    def chat_runner(query: str, model: str):
        captured["query"] = query
        return {
            "trajectory_id": "t2",
            "final_answer": "ok",
            "num_turns": 1,
            "success": True,
            "tool_calls": [],
            "messages": [],
            "tool_trace": [],
        }

    def multi_runner(query: str, model: str):
        return {
            "query": query,
            "num_targets": 0,
            "analyses": [],
            "report": "ok",
            "worklog": [],
        }

    svc = AgentQueryService(default_model="rule-based", chat_runner=chat_runner, multi_runner=multi_runner)
    svc.execute(
        "现在再看一次",
        conversation_history=[
            {"query": "上一轮问题", "chat": {"final_answer": "上一轮答案"}},
        ],
    )

    assert "以下是最近对话上下文" in captured["query"]
    assert "当前问题: 现在再看一次" in captured["query"]


def test_trajectory_payload_contains_react_and_reasoning_summary() -> None:
    svc = AgentQueryService(default_model="rule-based")
    trajectory = AgentTrajectory(
        trajectory_id="traj-x",
        prompt="测试",
        messages=[
            AgentMessage(role="system", content="sys"),
            AgentMessage(role="user", content="q"),
            AgentMessage(
                role="assistant",
                content="先搜索股票代码",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(
                            name="search_stock_by_name",
                            arguments=json.dumps({"name": "茅台", "market": "all"}, ensure_ascii=False),
                        ),
                    )
                ],
            ),
            AgentMessage(
                role="tool",
                name="search_stock_by_name",
                tool_call_id="call_1",
                content=json.dumps({"results": [{"code": "600519", "market": "a_share"}]}, ensure_ascii=False),
            ),
            AgentMessage(role="assistant", content="最终回答"),
        ],
        tool_calls=[
            ToolCall(
                id="call_1",
                type="function",
                function=FunctionCall(
                    name="search_stock_by_name",
                    arguments=json.dumps({"name": "茅台", "market": "all"}, ensure_ascii=False),
                ),
            )
        ],
        final_output="最终回答",
        num_turns=2,
        success=True,
    )

    payload = svc._trajectory_to_payload(trajectory)
    assert payload["reasoning_summary"]
    assert len(payload["react_trace"]) >= 3
    assert payload["react_trace"][0]["type"] == "thought"
    assert any(x["type"] == "action" for x in payload["react_trace"])
    assert any(x["type"] == "observation" for x in payload["react_trace"])
    assert any(x["type"] == "response" for x in payload["react_trace"])
