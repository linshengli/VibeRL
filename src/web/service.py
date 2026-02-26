from __future__ import annotations

import json
import inspect
from typing import Any, Callable, Dict, List, Optional

from src.agentic import AgenticOrchestrator
from src.agent import StockAnalysisAgent
from src.common.utils import try_json_loads
from src.models.entities import AgentTrajectory


class AgentQueryService:
    def __init__(
        self,
        default_model: str = "rule-based",
        debug_proxy: Optional[str] = None,
        max_targets: int = 3,
        chat_runner: Optional[Callable[[str, str], Dict[str, Any]]] = None,
        multi_runner: Optional[Callable[[str, str], Dict[str, Any]]] = None,
    ) -> None:
        self.default_model = default_model
        self.debug_proxy = debug_proxy
        self.max_targets = max_targets
        self._chat_runner = chat_runner
        self._multi_runner = multi_runner

    def execute(
        self,
        query: str,
        model: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        rag_chunks: Optional[List[Dict[str, Any]]] = None,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        used_model = model or self.default_model
        def emit(event: Dict[str, Any]) -> None:
            if event_callback is not None:
                event_callback(event)

        contextual_query = self._compose_query_with_context(query, conversation_history or [], rag_chunks or [])
        emit({"type": "stage", "stage": "chat.start", "detail": {"model": used_model}})
        chat_payload = self._run_chat(contextual_query, used_model)
        emit({"type": "stage", "stage": "chat.done", "detail": {"num_turns": chat_payload.get("num_turns")}})

        warnings: List[Dict[str, str]] = []
        try:
            emit({"type": "stage", "stage": "multi.start", "detail": {"mode": "agentic"}})
            multi_payload = self._run_multi(query, used_model, event_callback=emit)
            emit({"type": "stage", "stage": "multi.done", "detail": {"num_targets": multi_payload.get("num_targets")}})
        except Exception:
            multi_payload = {
                "query": query,
                "num_targets": 0,
                "analyses": [],
                "report": "多 Agent 执行失败，已返回 chat 结果。",
                "worklog": [],
                "error": {
                    "code": "MULTI_AGENT_FAILED",
                    "message": "multi-agent 执行失败，请检查日志后重试",
                },
            }
            warnings.append(
                {
                    "code": "MULTI_AGENT_FAILED",
                    "message": "multi-agent 执行失败，已降级返回 chat 结果",
                }
            )
            emit({"type": "stage", "stage": "multi.failed", "detail": {"reason": "MULTI_AGENT_FAILED"}})
        return {
            "status": "ok",
            "model": used_model,
            "chat": chat_payload,
            "multi_agent": multi_payload,
            "warnings": warnings,
        }

    def _run_chat(self, query: str, model: str) -> Dict[str, Any]:
        if self._chat_runner is not None:
            return self._chat_runner(query, model)

        agent = StockAnalysisAgent(model=model, base_url=self.debug_proxy)
        trajectory = agent.run(query)
        return self._trajectory_to_payload(trajectory)

    def _run_multi(
        self,
        query: str,
        model: str,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        if self._multi_runner is not None:
            try:
                params = inspect.signature(self._multi_runner).parameters
                if "event_callback" in params:
                    return self._multi_runner(query, model, event_callback=event_callback)
            except (TypeError, ValueError):
                pass
            return self._multi_runner(query, model)

        orchestrator = AgenticOrchestrator(
            max_targets=self.max_targets,
            model=model,
            base_url=self.debug_proxy,
        )
        return orchestrator.run(query, event_callback=event_callback)

    def _compose_query_with_context(
        self,
        query: str,
        history: List[Dict[str, Any]],
        rag_chunks: List[Dict[str, Any]],
    ) -> str:
        if not history and not rag_chunks:
            return query

        # Keep a compact rolling context for multi-turn continuity.
        lines: List[str] = []
        if history:
            recent = history[-6:]
            lines.append("以下是最近对话上下文：")
            for idx, turn in enumerate(recent, start=1):
                q = str(turn.get("query", "")).strip()
                answer = str(
                    (turn.get("chat") or {}).get("final_answer")
                    or (turn.get("multi_agent") or {}).get("report")
                    or ""
                ).strip()
                if q:
                    lines.append(f"[{idx}] 用户: {q}")
                if answer:
                    lines.append(f"[{idx}] 助手: {answer[:300]}")

        if rag_chunks:
            lines.append("")
            lines.append("以下是知识库检索结果（仅作参考，若与实时工具冲突以实时数据为准）：")
            for idx, item in enumerate(rag_chunks[:5], start=1):
                doc_name = str(item.get("doc_name") or "unknown")
                text = str(item.get("text") or "").strip().replace("\n", " ")
                lines.append(f"[RAG{idx}] 来源: {doc_name} | 内容: {text[:320]}")
        lines.append("")
        lines.append(f"当前问题: {query}")
        return "\n".join(lines)

    def _trajectory_to_payload(self, trajectory: AgentTrajectory) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = []
        tool_trace: List[Dict[str, Any]] = []
        react_trace: List[Dict[str, Any]] = []
        thought_fragments: List[str] = []
        react_step = 0

        for msg in trajectory.messages:
            messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                    "name": msg.name,
                    "tool_call_id": msg.tool_call_id,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": try_json_loads(tc.function.arguments, tc.function.arguments),
                        }
                        for tc in (msg.tool_calls or [])
                    ],
                }
            )

            if msg.tool_calls:
                thought = str(msg.content or "").strip()
                if thought:
                    thought_fragments.append(thought)
                    react_step += 1
                    react_trace.append({"type": "thought", "step": react_step, "content": thought})
                for tc in msg.tool_calls:
                    react_step += 1
                    action_payload = {
                        "type": "action",
                        "step": react_step,
                        "tool": tc.function.name,
                        "arguments": try_json_loads(tc.function.arguments, tc.function.arguments),
                    }
                    tool_trace.append(action_payload)
                    react_trace.append(action_payload)
            if msg.role == "tool" and msg.name:
                react_step += 1
                observation_payload = {
                    "type": "observation",
                    "step": react_step,
                    "tool": msg.name,
                    "result": try_json_loads(msg.content or "", msg.content),
                }
                tool_trace.append(observation_payload)
                react_trace.append(observation_payload)
            if msg.role == "assistant" and not msg.tool_calls:
                content = str(msg.content or "").strip()
                if content:
                    react_step += 1
                    react_trace.append({"type": "response", "step": react_step, "content": content})

        if thought_fragments:
            reasoning_summary = " | ".join(x for x in thought_fragments if x)[:800]
        else:
            assistant_contents = [
                str(m.get("content", "")).strip()
                for m in messages
                if m.get("role") == "assistant" and str(m.get("content", "")).strip()
            ]
            reasoning_summary = " | ".join(assistant_contents[:2])[:800]

        return {
            "trajectory_id": trajectory.trajectory_id,
            "final_answer": trajectory.final_output,
            "num_turns": trajectory.num_turns,
            "success": trajectory.success,
            "tool_calls": [tc.function.name for tc in trajectory.tool_calls],
            "messages": messages,
            "tool_trace": tool_trace,
            "react_trace": react_trace,
            "reasoning_summary": reasoning_summary,
        }
