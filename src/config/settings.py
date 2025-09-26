"""Application-wide configuration helpers.

This module centralises environment lookups so that CLI runners and LangChain
pipelines can share consistent defaults.
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Runtime settings loaded from environment or `.env` file."""

    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    judge_model: str = Field(default="gpt-4o-mini", env="JUDGE_MODEL")
    openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")

    vector_store: str = Field(default="faiss", env="VECTOR_STORE")
    embeddings_model: str = Field(default="text-embedding-3-small", env="EMBEDDINGS_MODEL")

    pop_size: int = Field(default=6, env="POP_SIZE")
    steps: int = Field(default=30, env="STEPS")
    mutate_every: int = Field(default=10, env="MUTATE_EVERY")
    tie_reward: float = Field(default=0.5, env="TIE_REWARD")
    eval_batch: int = Field(default=2, env="EVAL_BATCH")

    class Config:
        env_file = (Path(__file__).resolve().parents[2] / ".env",)
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()  # type: ignore[arg-type]


settings = get_settings()
