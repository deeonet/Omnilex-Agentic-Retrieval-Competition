"""LLM loader for the local retrieval pipeline.

Tries to load a local GGUF model via llama-cpp-python first; falls back to the
OpenAI-compatible API if the GGUF file is not present.  Both backends expose the
same callable interface: ``response["choices"][0]["text"]``.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from constants import CONFIG, MODEL_PATH


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
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        stop: list[str] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, list[dict[str, str]]]:
        completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model or self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs,
        )
        text = completion.choices[0].message.content or ""
        return {"choices": [{"text": text}]}


def load_llm():
    """Load the LLM — local GGUF if available, otherwise API.

    Local path: ``MODEL_PATH / CONFIG["model_file"]``
    Falls back to environment variables API_KEY / API_BASE_URL / API_MODEL.
    """
    model_file = CONFIG.get("model_file", "")
    gguf_path = MODEL_PATH / model_file if model_file else None

    if gguf_path and gguf_path.exists():
        from llama_cpp import Llama

        print(f"Loading local GGUF: {gguf_path}")
        lm = Llama(
            model_path=str(gguf_path),
            n_ctx=CONFIG.get("n_ctx", 8192),
            n_threads=CONFIG.get("n_threads", 4),
            n_gpu_layers=CONFIG.get("n_gpu_layers", -1),
            verbose=False,
        )
        print("Local GGUF model loaded.")
        return lm

    # Fall back to remote API
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
