"""LLM-as-a-judge helpers specialised for RAG evaluation."""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Dict, Tuple

from jinja2 import Template

MAX_RETRY = 2


@dataclass
class JudgeResult:
    verdict: str
    payload: Dict

    @property
    def win_margin(self) -> float:
        total = self.payload.get("total", {})
        return abs(total.get("A", 0.0) - total.get("B", 0.0))


def build_messages(template_text: str, instruction: str, context: str, ans_a: str, ans_b: str) -> Tuple[str, str]:
    """Build system/user messages using the raw template text."""

    blocks = template_text.split("[USER]")
    if len(blocks) != 2:
        raise ValueError("Judge template must contain a single [USER] marker")
    system = blocks[0].replace("[SYSTEM]", "").strip()
    pair = [("A", ans_a), ("B", ans_b)]
    random.shuffle(pair)
    rendered = Template(blocks[1]).render(
        instruction=instruction,
        context=context,
        answer_a=pair[0][1],
        answer_b=pair[1][1],
    )
    rendered += f"\n\n<!--order:{pair[0][0]}{pair[1][0]}-->"
    return system, rendered


def _restore_order(raw_payload: Dict, order_marker: str) -> Dict:
    if len(order_marker) != 2:
        return raw_payload
    new_to_original = {"A": order_marker[0], "B": order_marker[1]}
    original_to_new = {v: k for k, v in new_to_original.items()}

    adjusted = json.loads(json.dumps(raw_payload))
    for block_key in ("total", "scores"):
        block = adjusted.get(block_key, {})
        if isinstance(block, dict):
            adjusted[block_key] = {
                original: block.get(new_label)
                for original, new_label in original_to_new.items()
                if new_label in block
            }
    verdict = adjusted.get("verdict")
    if verdict in new_to_original:
        adjusted["verdict"] = new_to_original[verdict]
    return adjusted


def judge_pairwise_rag(
    chat_client,
    judge_template_text: str,
    instruction: str,
    context: str,
    ans_a: str,
    ans_b: str,
    tie_th: float = 0.05,
) -> JudgeResult:
    """Compare two answers under a shared context and return the verdict."""

    system, user = build_messages(judge_template_text, instruction, context, ans_a, ans_b)
    order_marker = "AB"
    if "<!--order:" in user:
        order_marker = user.split("<!--order:")[-1].split("-->")[0]
        user = user.replace(f"<!--order:{order_marker}-->", "")

    last_error = None
    for _ in range(MAX_RETRY + 1):
        raw = chat_client.chat(system, user)["text"]
        try:
            data = json.loads(raw)
            data = _restore_order(data, order_marker)
            total = data.get("total", {})
            if abs(total.get("A", 0.0) - total.get("B", 0.0)) < tie_th:
                data["verdict"] = "tie"
            return JudgeResult(data.get("verdict", "tie"), data)
        except Exception as exc:  # pragma: no cover - network call is hard to test
            last_error = {"raw": raw, "error": str(exc)}
            time.sleep(0.2)
            continue
    payload = {"error": "judge parse failed", "detail": last_error}
    return JudgeResult("tie", payload)
