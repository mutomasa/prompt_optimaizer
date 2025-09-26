"""Command line entry-point for tournament-based optimisation."""
from __future__ import annotations

from pathlib import Path

from src.config.settings import settings
from src.llm.client import ChatClient, build_langchain_chat
from src.llm.judge import judge_pairwise_rag
from src.optimize.runner import load_dataset, run_tournament_optimization
from src.prompts.loader import load_prompt
from src.rag.retriever import build_retriever


def build_judge(judge_template: str):
    client = ChatClient(model=settings.judge_model)

    def _judge(instruction: str, context: str, ans_a: str, ans_b: str):
        return judge_pairwise_rag(client, judge_template, instruction, context, ans_a, ans_b)

    return _judge


def main() -> None:
    prompt_cfg = load_prompt("prompts/rag_prompt.yaml")
    dataset = load_dataset("data/dataset.jsonl")
    judge_template = Path("prompts/judge_prompt.txt").read_text(encoding="utf-8")
    judge_fn = build_judge(judge_template)

    llm = build_langchain_chat(temperature=0.0)
    retriever = build_retriever()

    ranking = run_tournament_optimization(
        cfg=prompt_cfg,
        dataset=dataset,
        judge_fn=judge_fn,
        llm=llm,
        retriever=retriever,
    )

    for idx, row in enumerate(ranking, start=1):
        print(f"#{idx} win_rate={row['win_rate']:.3f} games={row['games']} knobs={row['knobs']}")


if __name__ == "__main__":
    main()
