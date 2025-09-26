"""Baseline tournament-style evaluation for prompt knobs."""
from __future__ import annotations

import random
from typing import Any, Dict, Sequence

from src.llm.judge import JudgeResult


def tournament_round(
    population: Sequence[Dict[str, Any]],
    dataset: Sequence[Dict[str, Any]],
    pipeline_builder,
    judge_fn,
    eval_batch: int = 2,
    tie_reward: float = 0.5,
) -> list[Dict[str, Any]]:
    """Run a full tournament round over the population."""

    results = {
        idx: {"wins": 0.0, "games": 0, "knobs": knobs}
        for idx, knobs in enumerate(population)
    }
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            knobs_a, knobs_b = population[i], population[j]
            pipeline_a = pipeline_builder(knobs_a)
            pipeline_b = pipeline_builder(knobs_b)
            batch_size = min(eval_batch, len(dataset))
            batch = random.sample(dataset, k=batch_size)
            for item in batch:
                payload = {
                    "instruction": item["instruction"],
                    "query": item.get("query"),
                    "docs": item.get("docs"),
                }
                output_a = pipeline_a.invoke(payload)
                output_b = pipeline_b.invoke(payload)
                context = output_a.get("context") or output_b.get("context") or ""
                verdict = judge_fn(
                    item["instruction"],
                    context,
                    output_a["response"],
                    output_b["response"],
                )
                if not isinstance(verdict, JudgeResult):
                    raise TypeError("judge_fn must return JudgeResult")
                if verdict.verdict == "A":
                    results[i]["wins"] += 1
                elif verdict.verdict == "B":
                    results[j]["wins"] += 1
                else:
                    results[i]["wins"] += tie_reward
                    results[j]["wins"] += tie_reward
                results[i]["games"] += 1
                results[j]["games"] += 1
    ranking: list[Dict[str, Any]] = []
    for rec in results.values():
        games = rec["games"] or 1
        ranking.append({
            "knobs": rec["knobs"],
            "win_rate": rec["wins"] / games,
            "games": rec["games"],
        })
    ranking.sort(key=lambda row: row["win_rate"], reverse=True)
    return ranking
