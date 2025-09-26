# Prompt Optimizer (LangChain Ready)

LangChain/LangGraph パイプラインでも利用できる **YAML/TOML 管理 + LLM-as-a-judge** のプロンプト最適化フレームワークです。RAG シナリオ向けに citation / faithfulness を重視した評価ループを実装しています。

## セットアップ
```bash
uv pip install -r requirements.txt
cp .env.example .env  # OPENAI_API_KEY などを設定
```

`.env` には `OPENAI_API_KEY`, `LLM_MODEL`, `JUDGE_MODEL`, `VECTOR_STORE` などをセットしてください。FAISS を利用する場合は `scripts/ingest_docs.py` で `indexes/faiss` を作成します。

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

## 自動最適化までの手順
1. **依存関係のインストール**: `uv pip install -r requirements.txt` を実行し、OpenAI/LangChain 周りのパッケージを揃えます。
2. **環境変数の設定**: `.env.example` をコピーし、`OPENAI_API_KEY` や使用するモデル名 (`LLM_MODEL`, `JUDGE_MODEL`) を記載します。FAISS など外部リソースを使う場合は `VECTOR_STORE` や `EMBEDDINGS_MODEL` も調整してください。
3. **データセットの準備**: `data/dataset.jsonl` に最適化対象の instruction（必要なら query, context, gold）を JSON Lines 形式で用意します。RAG の場合は `query` を retriever に渡します。
4. **Retriever 構築 (任意)**: FAISS インデックスを使う場合は `uv run python scripts/ingest_docs.py docs.jsonl indexes/faiss` でベクトルストアを作成します。`VECTOR_STORE` を `faiss` 以外にすることで別ストアに切り替え可能です。
5. **最適化の実行**:
   - Thompson Sampling: `uv run python src/app/main_ts.py`
   - 総当たりトーナメント: `uv run python src/app/main_tournament.py`
   実行時に `.env` の設定が読み込まれ、LangChain パイプラインが `src/rag/chains.py` を通じて呼び出されます。
6. **結果の確認**: CLI 実行後に出力される勝率やベストノブ構成を確認します。必要に応じて `src/optimize/runner.py` のログを利用し、メトリクス集計 (`src/eval/metrics.py`) を活用してください。
7. **LangChain への組み込み (任意)**: `build_rag_chain` をノードとして組み込み、ベスト構成のノブ値をフローに適用します。

## 主要コマンド
- `uv run python src/app/main_ts.py` : Thompson Sampling でノブ探索。
- `uv run python src/app/main_tournament.py` : 総当たりトーナメント評価。
- `uv run python scripts/ingest_docs.py docs.jsonl indexes/faiss` : JSONL から FAISS を生成（`page_content` もしくは `context` フィールド必須）。

## judge / メトリクス
- `prompts/judge_prompt.txt` は RAG 用ルーブリック（正確性/引用整合/指示遵守/表記）を含みます。
- `src/llm/judge.py` で A/B をランダム順序に並び替え、JSON スキーマを厳格パース。僅差 (0.05 未満) は `tie` として扱います。
- `src/eval/metrics.py` で総合点や citation 判定をログ整形できます。

## 注意事項
- retriever に FAISS を使用する場合は `indexes/` ディレクトリが `.gitignore` で除外されていることを確認してください。
- OPENAI_API_KEY を含む認証情報は `.env` に設定し、コミットしないこと。
- LangChain 依存のため、`langchain-openai` / `langchain-community` を requirements に含めています。環境によっては `faiss-cpu` がインストールできない場合があるため、その際は代替のベクトルストアに切り替えてください。
