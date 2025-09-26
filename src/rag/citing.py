"""Citation helpers for RAG chains."""
from __future__ import annotations

import json
import re
from typing import Iterable, Optional, Sequence

from langchain_core.documents import Document

_CITATION_PATTERN = re.compile(r"\[\[src:([^\]]+)]]")


def build_context_with_ids(docs: Sequence[Document]) -> str:
    """Format retrieved documents into a context string with ID markers."""

    lines = []
    for doc in docs:
        doc_id = doc.metadata.get("doc_id") or doc.metadata.get("id") or "NA"
        text = doc.page_content.strip().replace("\n", " ")
        lines.append(f"[{doc_id}] {text}")
    return "\n".join(lines)


def extract_citation_ids(text: str) -> list[str]:
    """Extract citation identifiers from the answer body."""

    return _CITATION_PATTERN.findall(text)


def check_citations(response: str, allowed_ids: Optional[Iterable[str]] = None) -> dict:
    """Validate citation markers in the model response.

    Parameters
    ----------
    response:
        Model output. The function attempts to parse JSON structures first and
        falls back to raw text.
    allowed_ids:
        Collection of valid document identifiers. When provided, citations
        outside this set are flagged.
    """

    parsed = None
    text = response
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict) and "answer" in parsed:
            text = parsed["answer"]
        else:
            text = response
    except json.JSONDecodeError:
        parsed = None

    citations = extract_citation_ids(response if not parsed else json.dumps(parsed))
    allowed = set(allowed_ids or [])
    citation_ids_valid = True
    if allowed:
        citation_ids_valid = all(cid in allowed for cid in citations)
    all_claims_cited = bool(citations)

    return {
        "response": parsed or response,
        "citations": citations,
        "citation_ids_valid": citation_ids_valid,
        "all_claims_cited": all_claims_cited,
    }
