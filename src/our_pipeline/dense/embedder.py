"""Loading and querying the Qwen3 embedding model.

torch / sentence-transformers are imported lazily inside the functions so that
importing this module (e.g. via search_tools) stays cheap for BM25-only runs.
"""

from __future__ import annotations

import numpy as np

from our_pipeline.constants import CONFIG


def load_embedding_model(device: str | None = None, max_seq_length: int | None = None):
    """Construct the SentenceTransformer for the configured embedding model.

    Single construction point shared by the offline corpus build and runtime
    query embedding, so both sides are guaranteed the same model setup.

    Args:
        device: Torch device string; auto-detects CUDA when None.
        max_seq_length: Optional token cap per input (corpus build sets this
            per corpus; queries are short so the default is fine).

    Returns:
        Loaded SentenceTransformer.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        CONFIG["embed_model"],
        device=device,
        # transformers >= 5 expects "dtype" (not "torch_dtype") in model_kwargs.
        model_kwargs={"dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32},
        # Left padding is required for Qwen3's last-token pooling.
        tokenizer_kwargs={"padding_side": "left"},
    )
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    return model


class QueryEmbedder:
    """Embeds search queries, loading the model on first use.

    Embeddings are cached per query string: the agent re-issues identical
    queries across iterations (same pattern as the tools' translation caches).
    """

    def __init__(self):
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, query: str) -> np.ndarray:
        """Return the normalized query embedding as a float32 vector."""
        if query not in self._cache:
            if self._model is None:
                self._model = load_embedding_model()
            self._cache[query] = self._model.encode(
                query,
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
        return self._cache[query]


_QUERY_EMBEDDER: QueryEmbedder | None = None


def get_query_embedder() -> QueryEmbedder:
    """Shared lazy singleton so all dense tools reuse one loaded model."""
    global _QUERY_EMBEDDER
    if _QUERY_EMBEDDER is None:
        _QUERY_EMBEDDER = QueryEmbedder()
    return _QUERY_EMBEDDER
