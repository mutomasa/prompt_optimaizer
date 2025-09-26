"""High-level orchestration for optimisation loops."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from src.config.settings import settings
from src.eval.metrics import summarise_match
from src.llm.judge import JudgeResult
from src.optimize.bandit_ts import ArmState, mutate_knobs, select_top_arms
from src.optimize.tournament import tournament_round
from src.prompts.loader import PromptConfig
from src.rag.chains import build_rag_chain
from src.rag.retriever import build_retriever

Dataset = List[Dict[str, Any]]


def load_dataset(path: str | Path = "data/dataset.jsonl") -> Dataset:
    """Load the optimisation dataset (instruction/query pairs)."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows: Dataset = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def prefetch_contexts(retriever, dataset: Dataset) -> Dataset:
    """Attach retrieved documents to each dataset row for fairness."""

    prepared: Dataset = []
    for item in dataset:
        payload = {"instruction": item["instruction"], "query": item.get("query")}
        docs = retriever.invoke(payload)
        docs_list = list(docs) if docs else []
        prepared.append({**item, "docs": docs_list})
    return prepared


def sample_knobs(space: Dict[str, Sequence[Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Sample a knob configuration from the search space."""

    config = dict(defaults)
    for key, options in space.items():
        if not options:
            continue
        config[key] = random.choice(list(options))
    return config


def bootstrap_population(space: Dict[str, Sequence[Any]], defaults: Dict[str, Any], size: int) -> List[Dict[str, Any]]:
    """Initialise a population with the default knob plus random samples."""

    population = [dict(defaults)]
    while len(population) < size:
        population.append(sample_knobs(space, defaults))
    return population


def _prepare_pipeline_factory(cfg: PromptConfig, llm=None, retriever=None) -> Callable[[Dict[str, Any]], Any]:
    retriever = retriever or build_retriever()

    def factory(knobs: Dict[str, Any]):
        return build_rag_chain(cfg, knobs, llm=llm, retriever=retriever)

    return factory


def run_tournament_optimization(
    cfg: PromptConfig,
    dataset: Dataset,
    judge_fn: Callable[[str, str, str, str], JudgeResult],
    population_size: int | None = None,
    eval_batch: int | None = None,
    tie_reward: float | None = None,
    llm=None,
    retriever=None,
) -> List[Dict[str, Any]]:
    """Execute a tournament round and return ranking information."""

    population_size = population_size or settings.pop_size
    eval_batch = eval_batch or settings.eval_batch
    tie_reward = tie_reward if tie_reward is not None else settings.tie_reward
    retriever = retriever or build_retriever()
    dataset_with_docs = prefetch_contexts(retriever, dataset)
    factory = _prepare_pipeline_factory(cfg, llm=llm, retriever=retriever)
    population = bootstrap_population(cfg.knobs, cfg.defaults, population_size)

    ranking = tournament_round(
        population=population,
        dataset=dataset_with_docs,
        pipeline_builder=factory,
        judge_fn=judge_fn,
        eval_batch=eval_batch,
        tie_reward=tie_reward,
    )
    return ranking


def run_thompson_sampling(
    cfg: PromptConfig,
    dataset: Dataset,
    judge_fn: Callable[[str, str, str, str], JudgeResult],
    steps: int | None = None,
    pop_size: int | None = None,
    mutate_every: int | None = None,
    tie_reward: float | None = None,
    eval_batch: int | None = None,
    llm=None,
    retriever=None,
) -> Dict[str, Any]:
    """Run Thompson Sampling optimisation and return collected statistics."""

    steps = steps or settings.steps
    pop_size = pop_size or settings.pop_size
    mutate_every = mutate_every or settings.mutate_every
    tie_reward = tie_reward if tie_reward is not None else settings.tie_reward
    eval_batch = eval_batch or settings.eval_batch
    retriever = retriever or build_retriever()
    dataset_with_docs = prefetch_contexts(retriever, dataset)
    factory = _prepare_pipeline_factory(cfg, llm=llm, retriever=retriever)

    arms: List[ArmState] = []
    seen: List[Dict[str, Any]] = []

    def _add_arm(knobs: Dict[str, Any]) -> None:
        if knobs in seen:
            return
        arms.append(ArmState(knobs=knobs))
        seen.append(knobs)

    _add_arm(dict(cfg.defaults))
    while len(arms) < pop_size:
        _add_arm(sample_knobs(cfg.knobs, cfg.defaults))

    logs: List[Dict[str, Any]] = []

    for step in range(steps):
        contenders = select_top_arms(arms, k=min(2, len(arms)))
        if len(contenders) < 2:
            break
        arm_a, arm_b = contenders[:2]
        pipeline_a = factory(arm_a.knobs)
        pipeline_b = factory(arm_b.knobs)
        batch = random.sample(dataset_with_docs, k=min(eval_batch, len(dataset_with_docs)))
        for item in batch:
            payload = {
                "instruction": item["instruction"],
                "query": item.get("query"),
                "docs": item.get("docs"),
            }
            out_a = pipeline_a.invoke(payload)
            out_b = pipeline_b.invoke(payload)
            context = out_a.get("context") or out_b.get("context") or ""
            judge_result = judge_fn(item["instruction"], context, out_a["response"], out_b["response"])
            if not isinstance(judge_result, JudgeResult):
                raise TypeError("judge_fn must return JudgeResult")
            if judge_result.verdict == "A":
                reward_a, reward_b = 1.0, 0.0
            elif judge_result.verdict == "B":
                reward_a, reward_b = 0.0, 1.0
            else:
                reward_a = reward_b = tie_reward
            summary = summarise_match(judge_result.verdict, judge_result.payload, tie_reward)
            arm_a.record(reward_a, metadata=summary)
            arm_b.record(reward_b, metadata=summary)
            logs.append({
                "step": step,
                "instruction": item["instruction"],
                "arm_a": arm_a.knobs,
                "arm_b": arm_b.knobs,
                "judge": judge_result.payload,
                "reward_a": reward_a,
                "reward_b": reward_b,
            })
        if (step + 1) % mutate_every == 0:
            arms.sort(key=lambda arm: arm.win_rate)
            weakest = arms[0]
            best = max(arms, key=lambda arm: arm.win_rate)
            mutated = mutate_knobs(cfg.knobs, best.knobs)
            arms[arms.index(weakest)] = ArmState(knobs=mutated)
    arms.sort(key=lambda arm: arm.win_rate, reverse=True)
    return {
        "arms": arms,
        "logs": logs,
    }
