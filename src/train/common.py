from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple


def parse_messages_payload(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    return []


def extract_user_prompt(messages: Sequence[Dict[str, Any]]) -> str:
    for msg in messages:
        if str(msg.get("role", "")).strip() == "user":
            text = str(msg.get("content", "")).strip()
            if text:
                return text
    return ""


def extract_last_assistant_answer(messages: Sequence[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if str(msg.get("role", "")).strip() == "assistant":
            text = str(msg.get("content", "")).strip()
            if text:
                return text
    return ""


def compute_rule_reward(text: str) -> float:
    content = str(text or "").strip()
    if not content:
        return 0.0

    score = 0.0
    if len(content) >= 30:
        score += 0.15
    if any(ch.isdigit() for ch in content):
        score += 0.15
    if any(key in content for key in ("风险", "建议", "策略", "仓位", "止损")):
        score += 0.2
    if any(key in content for key in ("涨跌", "价格", "市值", "RSI", "MACD", "MA")):
        score += 0.2
    if "\n" in content:
        score += 0.1
    return max(0.0, min(1.0, score))


def compute_overlap_reward(prediction: str, reference: str) -> float:
    pred = str(prediction or "").strip()
    ref = str(reference or "").strip()
    if not pred or not ref:
        return 0.0

    pred_set = set(pred)
    ref_set = set(ref)
    if not pred_set or not ref_set:
        return 0.0
    return max(0.0, min(1.0, len(pred_set & ref_set) / len(ref_set)))


def normalize_group_rewards(
    rewards: Sequence[float],
    use_ngrpo: bool = False,
    virtual_max_reward: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[List[float], float]:
    vals = [float(r) for r in rewards]
    if not vals:
        return [], 0.0

    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    std = var ** 0.5

    if std <= eps and use_ngrpo:
        augmented = vals + [float(virtual_max_reward)]
        mean = sum(augmented) / len(augmented)
        var = sum((x - mean) ** 2 for x in augmented) / len(augmented)
        std = var ** 0.5

    if std <= eps:
        return [0.0 for _ in vals], std

    advantages = [(x - mean) / (std + eps) for x in vals]
    return advantages, std
