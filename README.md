# Prompt Optimizer (LangChain Ready)

LangChain/LangGraph パイプラインでも利用できる **YAML/TOML 管理 + LLM-as-a-judge** のプロンプト最適化フレームワークです。RAG シナリオ向けに citation / faithfulness を重視した評価ループを実装しています。

## セットアップ
```bash
uv pip install -r requirements.txt
cp .env.example .env  # OPENAI_API_KEY などを設定
```

## ディレクトリ構成
```
prompts/                # RAG 用テンプレート & judge プロンプト
src/
  app/                  # Thompson Sampling / Tournament CLI エントリ
  config/               # BaseSettings ラッパー
  eval/                 # judge 出力のメトリクス集計
  llm/                  # OpenAI クライアント & judge 実装
  optimize/             # バンディット & トーナメントロジック
  prompts/              # YAML/TOML ローダー & Jinja レンダラー
  rag/                  # Retriever, Chain, citation チェック
scripts/ingest_docs.py  # JSONL -> FAISS index 変換ユーティリティ
data/dataset.jsonl      # instruction/query コンビ
```

## 主要コマンド
- `uv run python src/app/main_ts.py` : Thompson Sampling でノブ探索。
- `uv run python src/app/main_tournament.py` : 総当たりトーナメント評価。
- `uv run python scripts/ingest_docs.py docs.jsonl indexes/faiss` : JSONL から FAISS を生成（`page_content` もしくは `context` フィールド必須）。

## LangChain パイプラインへの組み込み
`src/rag/chains.py` の `build_rag_chain` は `RunnableLambda` を返すため、LCEL で `pipeline = build_rag_chain(cfg, knobs)` のように組み込み可能です。Retriever を差し替えたい場合は `build_retriever()` を自前で用意して引数に渡してください。

## judge / メトリクス
- `prompts/judge_prompt.txt` は RAG 用ルーブリック（正確性/引用整合/指示遵守/表記）を含みます。
- `src/llm/judge.py` で A/B をランダム順序に並び替え、JSON スキーマを厳格パース。僅差 (0.05 未満) は `tie` として扱います。
- `src/eval/metrics.py` で総合点や citation 判定をログ整形できます。

## Thompson Sampling ループ概要
1. `src/optimize/runner.py` がデータセットを読み込み、Retriever で context をキャッシュ。
2. `ArmState` ごとに Beta 事後を持ち、`select_top_arms` で 2 アームを比較。
3. `run_thompson_sampling` が judge verdict を reward (=勝率) に変換し、一定ステップごとに突然変異 (`mutate_knobs`) を適用。

## 注意事項
- retriever に FAISS を使用する場合は `scripts/ingest_docs.py` で `indexes/faiss` を生成してください。
- OPENAI_API_KEY を含む認証情報は `.env` に設定し、コミットしないこと。
- LangChain 依存のため、`langchain-openai` / `langchain-community` を requirements に含めています。環境によっては C++ ビルドの必要があるため `faiss-cpu` がインストールできない場合は `requirements.txt` から除外するか、別ベクトルストアに切り替えてください。
