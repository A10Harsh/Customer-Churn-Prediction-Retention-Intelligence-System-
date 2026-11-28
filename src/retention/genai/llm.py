"""A small model factory; only the Google adapter is implemented."""

import os
from functools import lru_cache

from retention.genai.config import GenAIConfig


def make_llm(config: GenAIConfig):
    if config.provider != "google":
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is required")
    return _google_model(
        config.model,
        config.temperature,
        config.request_timeout,
        config.max_retries,
        config.max_output_tokens,
    )


@lru_cache(maxsize=8)
def _google_model(
    model: str, temperature: float, timeout: int, retries: int, tokens: int
):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_retries=retries,
        max_output_tokens=tokens,
    )
