プログラム仕様書（dspyなし構成）
1. 目的

プロンプトのバリエーション（テンプレート＋ノブ）を探索し、

評価データセットを使って ペアワイズ比較 で勝ち残りを決め、

最適なプロンプト設定（knobs）を自動的に見つける。

2. 機能概要
(1) プロンプト管理

YAML/TOML形式でテンプレートとノブ（tone, steps, bullets など）を定義

jinja2で {instruction}, {context}, {tone} 等をレンダリング

defaults 値で基準のプロンプトを指定

(2) データセット

dataset.jsonl に以下の形式で格納

{"id":"ex1","instruction":"返品ポリシーを教えて","context":"30日以内の未開封は全額返金…","gold":"30日以内未開封は全額、開封済は10%手数料"}


instruction と context が入力、gold が期待出力（任意）

(3) モデル実行

任意の LLM クライアント（例：OpenAI, vLLM, HuggingFace Transformers）を使う

温度 0、top_p 1（再現性重視）

出力は文字列（JSON形式を推奨）

(4) ペアワイズ評価

LLM-as-a-judge を使い、2つの出力を比較

Judge用のプロンプト（YAML/TXTで固定）に

役割規定（厳格な採点官）

ルーブリック（正確性・根拠・指示遵守・表記）

出力形式（JSONスキーマ）

提示順序をランダム化（順序バイアス対策）

スコアリング例：

正確性 0–5

根拠提示 0–3

指示遵守 0–2

表記 0–1

→ 合計点を0–1に正規化

勝敗は 高得点の方を勝ち、僅差（差<0.05）は tie

(5) 最適化フロー

初期候補：ノブをランダムにサンプリング

各候補の勝率（勝数 / 試合数）を計算

ラウンドごとに上位候補を残し、突然変異で次世代候補を作成

停止条件：ラウンド数到達、改善なし、またはコスト上限

3. アーキテクチャ
flowchart TD
    A[dataset.jsonl] --> B[Prompt Renderer (jinja2)]
    B --> C[LLM Executor]
    C --> D[Candidate Outputs]
    D --> E[Pairwise Judge (LLM-as-a-judge)]
    E --> F[Score Aggregator]
    F --> G[Bandit / Tournament Optimizer]
    G -->|次候補Knobs| B
    G --> H[Best Prompt Config]
    H --> I[(Logs/MLflow)]

4. モジュール仕様
4.1 prompt_loader.py

YAML/TOMLを読み込み、PromptConfigクラスに格納

render_prompt(cfg, instruction, context, knobs) でテンプレートを埋め込む

4.2 llm_client.py

OpenAI APIなどを叩く簡易クラス

chat(system, user) -> {"text": "...", "usage": {"tokens":123}}

4.3 judge.py

Judge用プロンプトを読み込み、候補A/Bを比較

JSONを厳格パース、範囲外なら再実行

出力: {"verdict":"A","scores":{...},"total":{"A":0.8,"B":0.7}}

4.4 optimizer.py

ランダム探索 + トーナメント選択

勝率が高い候補を保持

突然変異：1ノブだけ変更した新候補を生成

4.5 main.py

データセット読み込み

初期候補を生成

各ラウンドで全候補を総当たり評価

最終的にベストknobsを出力

5. 入出力仕様
入力

プロンプト定義: prompts/base_prompt.yaml または .toml

データセット: data/dataset.jsonl

Judge定義: judge/judge_prompt.txt

出力

標準出力: ラウンドごとの勝率とベストknobs

JSONログ: history.json に各候補・勝率・スコアを保存

最終成果物: best_prompt.yaml（defaultsを最適値で更新）

6. 評価指標

勝率: ペアワイズ比較での勝率

総合スコア: 各観点スコアの正規化合計

安定性: tie率の低さ

運用面: トークンコスト、レイテンシ

7. 将来拡張

バンディット（Thompson Sampling, UCB1）を導入して探索効率化

promptfooでrubricを外部管理（YAML編集だけで評価変更可能）

MLflow/Langfuseで実験追跡

Streamlit UIで可視化（ノブ編集・結果表・勝率グラフ）

## Rag用の拡張

1) 目的・非機能要件
目的

RAGパイプライン（Retriever + LLM）で使用するプロンプトテンプレート/ノブを、ペアワイズ評価（LLM-as-a-judge） で自動最適化。

faithfulness（文脈忠実性）と指示遵守を重視したルーブリック採点で、最終的に最良のプロンプト構成を確定。

非機能要件

再現性：Judge温度0、few-shot固定、A/B順序ランダム化、seed（可能なら）固定。

拡張性：RetrieverやLLMエンドポイント、バンディット戦略（Thompson/UCB/Elo）を差し替え可能。

可観測性：LangChain callbacks/Tracing（LangSmith/Langfuse）対応、実験ログ（JSON/MLflow）。

コスト制御：サンプル件数、POP/STEPS/TIE_REWARD で推論回数を制御。

2) ディレクトリ構成
rag-prompt-optimizer/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ data/
│  └─ dataset.jsonl                 # instruction, query など（RAG前提）
├─ prompts/
│  ├─ rag_prompt.yaml               # YAML管理（TOMLも可）
│  └─ judge_prompt.txt              # LLM-as-a-judge（固定）
├─ judge/
│  └─ judge_rubric.yaml             # 配点やアンカー（説明用）
├─ src/
│  ├─ config/
│  │  └─ settings.py                # env/定数
│  ├─ rag/
│  │  ├─ retriever.py               # LangChain retriever factory
│  │  ├─ chains.py                  # LCEL/標準Chains（RAG回答器）
│  │  └─ citing.py                  # 引用ID整合、faithfulnessチェック
│  ├─ prompts/
│  │  ├─ loader.py                  # YAML/TOML読込 & jinja2レンダ
│  │  └─ schema.py                  # JSON Schema検証（任意）
│  ├─ llm/
│  │  ├─ client.py                  # ChatOpenAI/ChatOllama など抽象化
│  │  └─ judge.py                   # ペアワイズjudge（LLM-as-a-judge）
│  ├─ optimize/
│  │  ├─ bandit_ts.py               # Thompson Sampling（アーム=knobs）
│  │  ├─ tournament.py              # 総当り（baseline）
│  │  └─ runner.py                  # 反復ループ（評価→更新）
│  ├─ eval/
│  │  └─ metrics.py                 # ルーブリック集計・RAG特化チェック
│  └─ app/
│     ├─ main_ts.py                 # TSで最適化実行エントリ
│     └─ main_tournament.py         # 総当り実行エントリ
└─ scripts/
   └─ ingest_docs.py                # ベクタDBへの取り込み（任意）

3) 設定・環境
requirements（例）
langchain>=0.2
langchain-community>=0.2
langchain-openai>=0.2
jinja2>=3.1
pyyaml>=6.0
toml>=0.10
pydantic>=2.8
tqdm>=4.66
openai>=1.50
jsonschema>=4.23
# ベクタDB
faiss-cpu>=1.8  # or chromadb, weaviate-client, pinecone-client 等

環境変数（.env.example）
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o-mini
OPENAI_BASE_URL=               # 互換API利用時
VECTOR_STORE=faiss             # "faiss" | "chroma" | ...
EMBEDDINGS_MODEL=text-embedding-3-small
POP_SIZE=6
STEPS=30
MUTATE_EVERY=10
TIE_REWARD=0.5
EVAL_BATCH=2                   # 1対戦で評価するサンプル件数

4) プロンプト管理（YAML/TOML）
prompts/rag_prompt.yaml（例）
task: "rag_qa_ja"
template:
  system: |
    あなたは厳格な事実重視アシスタントです。{tone}な文体で回答し、必ず引用ID [[src:...]] を付与してください。
    出力は必ずJSONで schema: {schema} に従うこと。根拠不明なら "不明" と明記。
  user: |
    # 質問
    {instruction}

    # 検索文脈（抜粋）
    {context}

    # ルール
    - 文脈外の知識は推測禁止
    - 各主張に [[src:<DOC_ID>]] を付ける
  constraints: |
    - 言語は厳密に日本語
    - 箇条書きは最大 {bullets} 個
    - 推論ステップは {steps} 個以内
knobs:
  tone: ["丁寧", "端的", "厳格"]
  bullets: [0, 3, 5]
  steps: [3, 5, 7]
defaults:
  tone: "丁寧"
  bullets: 3
  steps: 5
schema: |
  {"type":"object","properties":{
     "answer":{"type":"string"},
     "citations":{"type":"array","items":{"type":"string"}}
   },"required":["answer","citations"]}


context は Retriever から入れる。各チャンクには id を付与して [[src:ID]] で参照可能にする。

TOML 版も可。loader は拡張子で分岐。

5) RAGパイプライン（LangChain）
5.1 Retriever

embeddings：langchain-openai. OpenAIEmbeddings（または SentenceTransformersEmbedding）

vectorstore：FAISS/Chroma/Pineconeなど選択可

retriever：as_retriever(search_type="similarity", k=K, score_threshold=...)

5.2 Chain（LCEL）

構成：

入力: {"instruction": str}

Retriever → Docs（id, text）

ContextBuilder：Docsを {context} 文字列（ID付きフォーマット）へ整形

PromptRenderer：YAMLテンプレ & knobs で system/user を jinja2 レンダ

Chat（LLM）

PostProcess：JSONパース、Schema検証、引用ID抽出、引用整合チェック（src/citing.py）

LCEL例（擬似）：

from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_openai import ChatOpenAI
from src.prompts.loader import load_prompt, render_prompt
from src.rag.retriever import build_retriever
from src.rag.citing import build_context_with_ids, check_citations

cfg = load_prompt("prompts/rag_prompt.yaml")
llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=0)

def pipeline(knobs):
    retriever = build_retriever()  # returns LC retriever
    chain = (
        RunnableMap({
            "instruction": lambda x: x["instruction"],
            "docs": retriever
        })
        | RunnableLambda(lambda x: {
            "instruction": x["instruction"],
            "context": build_context_with_ids(x["docs"])  # -> string with [[src:doc_id]]
        })
        | RunnableLambda(lambda x: {
            "rendered": render_prompt(cfg, x["instruction"], x["context"], knobs)
        })
        | RunnableLambda(lambda x: {
            "response": llm.invoke([
                {"role": "system", "content": x["rendered"]["system"]},
                {"role": "user", "content": x["rendered"]["user"]}
            ]).content
        })
        | RunnableLambda(lambda x: check_citations(x["response"], allowed_ids=None))
    )
    return chain

6) Judge（LLM-as-a-judge：RAG特化）
judge_prompt.txt（要点）

役割規定：採点官、生成禁止、温度0

ルーブリック：

正確性 (0–5)：与えた文脈への忠実性（外部知識の追加は減点）

根拠提示 (0–3)：[[src:ID]] の整合、主張ごとの引用有無

指示遵守 (0–2)：JSON/語調/出力仕様

表記 (0–1)：日本語/記法

追加チェック（出力JSONに含める）：

citation_ids_valid: bool（回答内のIDが文脈ID集合に含まれるか）

all_claims_cited: bool（主要文が引用付きか・ヒューリスティックでも可）

judge.py（インタフェース）
def judge_pairwise_rag(chat_client, judge_template_text, instruction, ctx_a, ans_a, ctx_b, ans_b, tie_th=0.05):
    """
    ctx_a/ctx_b を同一にするのが基本（公平性）。
    JSONパース、範囲/整合チェック、僅差tie。
    return verdict["A"|"B"|"tie"], detail_json
    """


※ 基本は 同一の context 下で A/B を比較します（retrieverに確率的要素がある場合は固定seedまたはキャッシュ）。

7) Optimizer（Thompson Sampling 推奨）
7.1 Arm定義

Arm = knobs（例：tone, steps, bullets, cite_style など）

統計：wins, plays, Beta(α,β) 事後（TS用）

7.2 ループ仕様（main_ts.py）

初期：POP_SIZE 個の knobs をランダムサンプル

1ステップ：

TSで 上位2アーム を選出

EVAL_BATCH 件取り出し（datasetの「instruction」）

各 instruction について 同一のRetriever文脈を使い A/B を生成 → Judge

多数決で verdict（A/B/tie）

record_result()（win=1, loss=0, tie=TIE_REWARD）

探索：MUTATE_EVERY ステップ毎に、ベストknobsから突然変異した新アームで最下位を置換

停止：STEPS 到達 or 改善停滞

8) データセット仕様（RAG向け）

data/dataset.jsonl

{"id":"q1","instruction":"製品Aの返品条件は？","query":"返品 ポリシー 製品A"}
{"id":"q2","instruction":"営業時間を教えて","query":"営業時間 店舗A"}


実行時：query を retriever に渡して context を作成 → pipeline で生成

ただし A/B公平性のため、同一 instruction に対しては retriever 結果を固定キャッシュして使い回す（両者同じ context）。

9) メトリクス・ログ
メトリクス

勝率（TSは事後平均も参照）

ルーブリック合計（0–1正規化）

faithfulness 補助指標：citation_ids_valid, all_claims_cited, claims_outside_context（ヒューリスティック）

コスト/レイテンシ（LLM usage & 時間）

ログ

JSON出力：各ステップの summary・Arm統計・ベストknobs

Tracing：LangChain callbacks（LangSmith/Langfuse）有効化

（任意）MLflow：パラメータ（knobs）、メトリクス、アーティファクト（上位回答・失敗例）

10) 例：主要インタフェース（擬似コード）
src/rag/retriever.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def build_retriever():
    # すでに FAISS index が構築済みという前提。未構築なら scripts/ingest_docs.py を用意
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"))
    vs = FAISS.load_local("indexes/faiss", embeddings, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_type="similarity", k=4)

src/rag/citing.py
def build_context_with_ids(docs):
    # docs: List[Document] with metadata {"doc_id": "..."}
    lines = []
    for d in docs:
        doc_id = d.metadata.get("doc_id", "NA")
        lines.append(f"[{doc_id}] {d.page_content}")
    return "\n".join(lines)

def check_citations(resp: str, allowed_ids=None):
    # 返信中の [[src:ID]] を抽出し、allowed_ids と照合（allowed_ids を渡さない場合はスキップ可）
    # JSONパース→ "citations" を拾う設計でもOK
    return {"response": resp, "citations_ok": True}

src/llm/judge.py
def judge_pairwise_rag(chat_client, judge_template_text, instruction, context, ans_a, ans_b, tie_th=0.05):
    # judge_template_text には [SYSTEM] ... [USER] ... を含めておく（前回提案フォーマット）
    # A/B順はランダム化して user を生成。JSON出力を厳格パース、僅差tie。
    return verdict, detail_json

src/optimize/runner.py（TS運用）
def run_ts(dataset, knobs_space, pipeline_builder, judge, steps, mutate_every, tie_reward):
    # pipeline_builder(knobs) -> LCEL chain
    # dataset item: {"instruction":..., "query":...}
    # retriever結果は instruction ごとにキャッシュして A/B で共通化
    ...

11) 安全・品質ガード（RAG特化）

引用必須：テンプレ側で [[src:ID]] を強制、judgeで未引用を減点。

外部知識の抑制：systemに「文脈外の推測禁止」を明示。

JSON Schema検証：生成直後にバリデーション、失敗なら1回リトライ（温度0）。

順序バイアス対策：A/B提示順ランダム化（内部マッピングで元A/Bへ戻す）。

僅差tie：|A-B|<0.05 は tie 扱い。

データリーク防止：judgeには与えたcontext以外の情報を見せない（instruction + 同一context のみ）。

12) 運用・拡張

バンディット差し替え：optimize/bandit_ts.py → bandit_ucb.py を追加可能。dueling bandit（Elo/Bradley–Terry）も選択肢。

評価の二段構え：先に ルールベース検査（JSON/引用整合） を実行し、同点時のみ LLM Judge を使う節約モード。

A/B 本番運用：最良knobsを本番Chainに反映し、10%トラフィックを探索に回す運用（online-TS）。

LangSmith/Langfuse：retrieval・生成・judge の各ステップをトレースし、失敗ケースをギャラリー化。