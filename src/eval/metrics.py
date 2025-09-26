"""Evaluation helpers for collecting optimisation metrics."""
from __future__ import annotations

from typing import Dict


def compute_reward(verdict: str, tie_reward: float) -> float:
    """Convert a judge verdict into a reward signal for the optimiser."""

    if verdict == "A":
        return 1.0
    if verdict == "B":
        return 0.0
    return tie_reward


def collapse_scores(judge_payload: Dict) -> Dict[str, float]:
    """Extract scalar metrics from the judge payload."""

    scores = judge_payload.get("scores", {})
    total = judge_payload.get("total", {})
    return {
        "accuracy": float(scores.get("accuracy", 0.0)),
        "grounding": float(scores.get("grounding", 0.0)),
        "instruction_following": float(scores.get("instruction_following", 0.0)),
        "presentation": float(scores.get("presentation", 0.0)),
        "total_a": float(total.get("A", 0.0)),
        "total_b": float(total.get("B", 0.0)),
        "citation_ids_valid": float(judge_payload.get("citation_ids_valid", False)),
        "all_claims_cited": float(judge_payload.get("all_claims_cited", False)),
    }


def summarise_match(verdict: str, detail: Dict, tie_reward: float) -> Dict:
    """Return a compact match summary suitable for logging."""

    summary = collapse_scores(detail)
    summary.update({
        "verdict": verdict,
        "reward": compute_reward(verdict, tie_reward),
    })
    return summary
