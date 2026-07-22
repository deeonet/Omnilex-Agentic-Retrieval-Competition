"""Minimal OpenAI-compatible embedding server backed by sentence-transformers.

Exposes POST /v1/embeddings — same contract as OpenAI's API so GraphRAG
(which talks to the API through LiteLLM) works without modification.

Usage:
    python scripts/embedding_server.py \
        --model models/multilingual-e5-large \
        --port 8002
"""
import argparse
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── Global model holder ────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None
_model_name: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_name
    args = _parsed_args
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model from {args.model} on {device}...")
    _model = SentenceTransformer(args.model, device=device)
    _model_name = Path(args.model).name
    print(f"Model ready: {_model_name}")
    yield


app = FastAPI(lifespan=lifespan)


# ── Request / response schemas (OpenAI-compatible) ────────────────────────────

class EmbeddingRequest(BaseModel):
    input: list[str] | str
    model: str | None = None


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingObject]
    model: str
    usage: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": _model_name}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": _model_name, "object": "model"}],
    }


@app.post("/v1/embeddings")
def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    texts = request.input if isinstance(request.input, list) else [request.input]

    # multilingual-e5 models need a "query: " prefix for query embeddings.
    # GraphRAG sends document-style text; use "passage: " prefix.
    prefixed = ["passage: " + t for t in texts]

    embeddings: np.ndarray = _model.encode(
        prefixed,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )

    data = [
        EmbeddingObject(index=i, embedding=emb.tolist())
        for i, emb in enumerate(embeddings)
    ]

    return EmbeddingResponse(
        data=data,
        model=_model_name,
        usage={"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": 0},
    )


# ── Entry point ───────────────────────────────────────────────────────────────

_parsed_args: argparse.Namespace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to sentence-transformers model")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", default="127.0.0.1")
    _parsed_args = parser.parse_args()

    uvicorn.run(app, host=_parsed_args.host, port=_parsed_args.port, log_level="info")
