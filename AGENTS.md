# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains production code: `app/` (CLI entrypoints), `config/` (Pydantic settings), `llm/` (OpenAI + judge), `prompts/` (YAML/TOML loader and schema helpers), `rag/` (retriever, chains, citation checks), `optimize/` (TS and tournament logic), and `eval/` (metric reducers). Keep new runtime logic adjacent to these modules so LangChain pipelines import cleanly.
- `prompts/` stores editable templates: `rag_prompt.yaml` for generation and `judge_prompt.txt` for LLM-as-a-judge. Adjust both alongside code changes to maintain schema/constraint parity.
- `data/dataset.jsonl` holds optimisation examples (`id`, `instruction`, optional `query/context/gold`). Use newline-delimited JSON; curate lightweight fixtures for tests.
- `scripts/ingest_docs.py` builds FAISS indexes from JSONL corpora. Place derived artifacts under `indexes/` (ignored) so retrievers stay deterministic.

## Build, Test, and Development Commands
- `uv pip install -r requirements.txt` syncs dependencies (LangChain, OpenAI SDK, jsonschema, faiss-cpu).
- `uv run python src/app/main_ts.py` and `uv run python src/app/main_tournament.py` execute optimisation rounds; ensure `.env` carries `OPENAI_API_KEY` and model names.
- `uv run python -m pytest` (once tests exist) should stub LLM/network calls via fixtures; prefer `-k` expressions for targeted runs.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indent, snake_case for functions/variables, PascalCase for classes, and type hints on public APIs.
- Keep retriever/chain utilities side-effect free; expose dependency-injected builders instead of hard-coding LangChain components.
- Document new knob names and prompt schema adjustments in `spec.md` and `prompts/rag_prompt.yaml` to keep config discoverable.

## Testing Guidelines
- Build `tests/` around pytest. Seed `random` when exercising tournament/TS flows to guarantee reproducibility.
- Mock `ChatClient` and LangChain runnables so tests remain offline; assert JSON parsing and tie-handling paths in `src/llm/judge.py`.
- Add schema validation cases (`src/prompts/schema.py`) and citation heuristics (`src/rag/citing.py`) to prevent regressions when prompts evolve.

## Commit & Pull Request Guidelines
- Use imperative commit subjects (`Add citation validator`) with concise bodies describing affected modules/datasets.
- Reference tracking issues in PR descriptions, attach sample judge outputs or optimiser logs, and summarize dataset/prompt diffs.
- Update documentation (`README.md`, `AGENTS.md`, optionally `spec.md`) whenever behaviour, CLI interfaces, or configuration expectations change.

## LangChain Integration Tips
- Reuse `build_rag_chain` for LCEL/Graph nodes; pass a shared retriever via dependency injection to keep contexts aligned between A/B candidates.
- When embedding inside larger pipelines, surface knob metadata in the chain inputs/outputs so upstream schedulers can introspect current settings.
- For alternative vector stores, swap `build_retriever()` with a custom Runnable that returns `Document` sequences but retains the cached-context contract.
