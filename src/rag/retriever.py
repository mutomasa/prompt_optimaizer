"""Retriever factories for RAG optimisation pipelines."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.config.settings import settings


def _load_faiss_retriever():  # pragma: no cover - optional dependency
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    index_path = Path("indexes/faiss")
    if not index_path.exists():
        raise FileNotFoundError("FAISS index not found at indexes/faiss")
    embeddings = OpenAIEmbeddings(model=settings.embeddings_model)
    vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_type="similarity", k=4)
    return RunnableLambda(lambda payload: retriever.invoke(payload.get("query") or payload.get("instruction")))


@lru_cache(maxsize=1)
def _load_dataset_contexts() -> list[Document]:
    dataset_path = Path("data/dataset.jsonl")
    if not dataset_path.exists():
        return []
    contexts: list[Document] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            context = item.get("context")
            if not context:
                continue
            doc_id = item.get("id") or f"item-{idx}"
            contexts.append(Document(page_content=context, metadata={"doc_id": doc_id}))
    return contexts


def _fallback_retriever_payload(payload: dict) -> Iterable[Document]:
    contexts = _load_dataset_contexts()
    if not contexts:
        instruction = payload.get("instruction", "?")
        doc = Document(page_content=f"Context unavailable for: {instruction}", metadata={"doc_id": "fallback"})
        return [doc]
    query = payload.get("instruction") or payload.get("query")
    if not query:
        return contexts[:3]
    matches = [doc for doc in contexts if query in doc.page_content]
    return matches or contexts[:3]


def build_retriever():
    """Return a retriever suitable for LangChain pipelines."""

    if settings.vector_store.lower() == "faiss":
        try:
            return _load_faiss_retriever()
        except Exception:
            pass
    return RunnableLambda(_fallback_retriever_payload)
