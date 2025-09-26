"""LangChain pipeline builders for RAG prompt optimisation."""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from src.llm.client import build_langchain_chat
from src.prompts.loader import PromptConfig, render_prompt
from src.rag.citing import build_context_with_ids, check_citations
from src.rag.retriever import build_retriever


def build_rag_chain(
    cfg: PromptConfig,
    knobs: Dict[str, Any],
    llm=None,
    retriever=None,
) -> RunnableLambda:
    """Create a LangChain runnable that executes the RAG pipeline."""

    retriever = retriever or build_retriever()
    llm = llm or build_langchain_chat(temperature=0.0)

    def _pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
        instruction = payload["instruction"]
        query = payload.get("query", instruction)
        docs = payload.get("docs")
        if docs is None:
            retriever_input = {**payload, "instruction": instruction, "query": query}
            docs = retriever.invoke(retriever_input)
        docs = list(docs) if docs else []
        context = build_context_with_ids(docs)
        rendered = render_prompt(cfg, instruction, context, knobs)
        user_content = rendered["user"]
        if rendered.get("constraints"):
            user_content = f"{user_content}\n\n{rendered['constraints']}"
        messages = [
            SystemMessage(content=rendered["system"]),
            HumanMessage(content=user_content),
        ]
        result = llm.invoke(messages)
        text = getattr(result, "content", None) or getattr(result, "text", None) or str(result)
        doc_ids = [doc.metadata.get("doc_id") for doc in docs]
        citation_report = check_citations(text, allowed_ids=doc_ids)
        return {
            "response": text,
            "citations": citation_report,
            "prompt": rendered,
            "context": context,
            "docs": docs,
        }

    return RunnableLambda(_pipeline)
