from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.core import StockAnalysisAgent
from src.models.entities import AgentMessage


def _format_assistant_message(message: AgentMessage) -> Optional[str]:
    if message.role != "assistant":
        return None
    if message.tool_calls:
        lines = []
        if message.content:
            lines.append(f"[Thought] {message.content}")
        for call in message.tool_calls:
            lines.append(f"[Action] {call.function.name}({call.function.arguments})")
        return "\n".join(lines)
    if message.content:
        return f"[Final Answer] {message.content}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stock analysis ReAct agent")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--model", default="rule-based", help="Model name (rule-based by default)")
    parser.add_argument("--max-steps", type=int, default=10, help="Max tool-calling steps")
    parser.add_argument(
        "--debug-proxy",
        default=None,
        help="Optional OpenAI-compatible base_url (e.g. http://localhost:8080/v1)",
    )
    args = parser.parse_args()

    agent = StockAnalysisAgent(model=args.model, base_url=args.debug_proxy)
    trajectory = agent.run(user_query=args.query, max_steps=args.max_steps)

    for message in trajectory.messages:
        rendered = _format_assistant_message(message)
        if rendered:
            print(rendered)
        elif message.role == "tool" and message.name:
            print(f"[Observation:{message.name}] {message.content}")


if __name__ == "__main__":
    main()
