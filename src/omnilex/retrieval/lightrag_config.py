"""LightRAG wiring for the Swiss legal citation pipeline.

Provides the LLM + embedding adapters (pointed at locally-served models) and a
``make_lightrag`` factory that builds and initialises a ``LightRAG`` instance.

Model serving (started by scripts/run_lightrag_index.sh):
  - Completion: Qwen served by vLLM at ${LIGHTRAG_LLM_BASE_URL} (OpenAI-compatible)
  - Embedding:  multilingual-e5-large via scripts/embedding_server.py at
                ${LIGHTRAG_EMBED_BASE_URL} (OpenAI-compatible /v1/embeddings)

LightRAG stores its index as plain local files (JSON KV + NanoVectorDB +
NetworkX graph) under ``working_dir`` — no parquet / pyarrow involved.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Endpoints (default to the local servers; override via .env) ───────────────
LLM_BASE_URL = os.getenv("LIGHTRAG_LLM_BASE_URL", "http://127.0.0.1:8001/v1")
LLM_MODEL = os.getenv("LIGHTRAG_LLM_MODEL", "Qwen2.5-14B-Instruct")
LLM_API_KEY = os.getenv("LIGHTRAG_LLM_API_KEY", "EMPTY")

EMBED_BASE_URL = os.getenv("LIGHTRAG_EMBED_BASE_URL", "http://127.0.0.1:8002/v1")
EMBED_MODEL = os.getenv("LIGHTRAG_EMBED_MODEL", "multilingual-e5-large")
EMBED_API_KEY = os.getenv("LIGHTRAG_EMBED_API_KEY", "EMPTY")
EMBED_DIM = int(os.getenv("LIGHTRAG_EMBED_DIM", "1024"))   # multilingual-e5-large
EMBED_MAX_TOKENS = int(os.getenv("LIGHTRAG_EMBED_MAX_TOKENS", "512"))


async def llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    """LightRAG-compatible completion func backed by the local Qwen (vLLM) server."""
    # Imported lazily so this module is importable without lightrag installed.
    from lightrag.llm.openai import openai_complete_if_cache

    # Strip kwargs LightRAG injects that the local OpenAI server does not accept.
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        **kwargs,
    )


async def _embed(texts: list[str]) -> np.ndarray:
    """Embed texts via the OpenAI-compatible e5 server (scripts/embedding_server.py)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY)
    resp = await client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def make_embedding_func():
    """Build the LightRAG EmbeddingFunc wrapper for the e5 server."""
    from lightrag.utils import EmbeddingFunc

    return EmbeddingFunc(
        embedding_dim=EMBED_DIM,
        max_token_size=EMBED_MAX_TOKENS,
        func=_embed,
    )


async def make_lightrag(working_dir: str | Path, **overrides: Any):
    """Construct and initialise a LightRAG instance for the given working dir.

    Laws articles are short, so we set a large ``chunk_token_size`` (default
    8192) to keep one article = one chunk → 1:1 citation↔chunk mapping. Override
    any LightRAG kwarg via ``overrides``.
    """
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import initialize_pipeline_status

    working_dir = str(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    params: dict[str, Any] = dict(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        llm_model_name=LLM_MODEL,
        embedding_func=make_embedding_func(),
        chunk_token_size=8192,
        chunk_overlap_token_size=0,
    )
    params.update(overrides)

    rag = LightRAG(**params)
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag
