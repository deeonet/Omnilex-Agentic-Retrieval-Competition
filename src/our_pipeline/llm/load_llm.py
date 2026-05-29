"""OpenAI-compatible LLM adapter for the local retrieval pipeline.

The agent code was written against ``llama-cpp-python``, whose model object is
called like a function and returns ``{"choices": [{"text": "..."}]}``.  This
module keeps that small interface intact while sending requests to the API used
in ``test.py``.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_BASE_URL = "https://chat-ai.academiccloud.de/v1"
DEFAULT_MODEL = "qwen3.5-27b"


class OpenAICompatibleLLM:
    """Callable adapter that mimics the llama-cpp completion response shape."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """Create an API-backed LLM.

        Args:
            api_key: API key for the OpenAI-compatible service.
            base_url: Base URL of the service.
            model: Model name to request.
        """
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, list[dict[str, str]]]:
        """Generate text from a raw prompt using chat completions.

        Args:
            prompt: Full agent prompt/conversation string.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop: Optional stop sequences.
            **kwargs: Additional API arguments forwarded to chat completions.

        Returns:
            A llama-cpp-style response with generated text at
            ``response["choices"][0]["text"]``.
        """
        completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs,
        )
        text = completion.choices[0].message.content or ""
        return {"choices": [{"text": text}]}


def load_llm() -> OpenAICompatibleLLM:
    """Load the API-backed LLM using values from ``.env``.

    Expected environment variables:
        API_KEY: Required API key.
        API_BASE_URL: Optional API base URL.
        API_MODEL: Optional model name.

    Returns:
        Configured API-backed LLM adapter.

    Raises:
        RuntimeError: If ``API_KEY`` is missing.
    """
    load_dotenv()

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Missing API_KEY. Add API_KEY=... to your .env file.")

    base_url = os.getenv("API_BASE_URL", DEFAULT_BASE_URL)
    model = os.getenv("API_MODEL", DEFAULT_MODEL)

    print(f"Loading API model: {model}")
    llm_client = OpenAICompatibleLLM(api_key=api_key, base_url=base_url, model=model)
    print("API model client loaded successfully!")

    return llm_client


llm = load_llm()
