"""Utility script to build a FAISS index from JSONL documents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config.settings import settings


def load_documents(path: Path) -> list[Document]:
    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            content = data.get("page_content") or data.get("text") or data.get("context")
            if not content:
                continue
            doc_id = data.get("id") or data.get("doc_id") or str(len(docs))
            docs.append(Document(page_content=content, metadata={"doc_id": doc_id}))
    return docs


def build_index(input_path: Path, output_dir: Path) -> None:
    docs = load_documents(input_path)
    if not docs:
        raise RuntimeError("No documents found to index")
    embeddings = OpenAIEmbeddings(model=settings.embeddings_model)
    vectorstore = FAISS.from_documents(docs, embeddings)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index for RAG retriever")
    parser.add_argument("input", type=Path, help="Path to JSONL file containing documents")
    parser.add_argument("output", type=Path, help="Directory to store the FAISS index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(args.input, args.output)


if __name__ == "__main__":
    main()
