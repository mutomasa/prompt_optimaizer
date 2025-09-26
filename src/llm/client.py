"""Lightweight OpenAI client wrappers used across the project."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

from openai import OpenAI

from src.config.settings import settings


class ChatClient:
    """Minimal chat client around the OpenAI SDK."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        model = model or settings.llm_model
        api_key = api_key or settings.openai_api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key, base_url=base_url or settings.openai_base_url)
        self.model = model

    def chat(self, system: str, user: str) -> Dict:
        """Send a system/user message pair and return the response payload."""

        return self.invoke_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

    def invoke_messages(self, messages: Iterable[Dict[str, str]]) -> Dict:
        """Invoke the chat completion API with arbitrary messages."""

        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=list(messages),
        )
        choice = response.choices[0].message
        usage = response.usage
        return {
            "text": choice.content,
            "usage": {
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            },
        }


def build_langchain_chat(temperature: float = 0.0):
    """Return a LangChain-compatible chat model if langchain is installed."""

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("langchain-openai is not installed") from exc

    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
