"""CrossEncoder reranking of retrieved citations for precision.

The recall_union retriever returns a broad, recall-first union (~160 citations) of which
only ~20-40 are relevant. This module re-scores each candidate against the query with a
Qwen3-Reranker-4B CrossEncoder and keeps only the citations it is confident about, raising
precision (and F1) at a modest recall cost.

The pipeline passes citation *strings* only, so reranking needs each citation's full
document text. That text is recovered from the in-memory BM25 indices via
``make_citation_text_lookup`` (full, untruncated text -- unlike the dense excerpts).

torch / sentence-transformers are imported lazily inside the functions so that importing
this module (e.g. via predictions) stays cheap for BM25-only runs.
"""

from __future__ import annotations

from typing import Callable, Iterable

from our_pipeline.constants import CONFIG


def load_reranker(device: str | None = None):
    """Construct the CrossEncoder for the configured reranker model.

    Mirrors ``dense.embedder.load_embedding_model``: lazy heavy imports, CUDA
    auto-detection, and bfloat16 on GPU.

    Args:
        device: Torch device string; auto-detects CUDA when None.

    Returns:
        Loaded CrossEncoder.
    """
    import torch
    from sentence_transformers import CrossEncoder

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return CrossEncoder(
        CONFIG["reranker_model"],
        device=device,
        max_length=CONFIG.get("reranker_max_length", 1024),
        # transformers >= 5 expects "dtype" (not "torch_dtype") in model_kwargs.
        model_kwargs={"dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32},
    )


def make_citation_text_lookup(*indices) -> Callable[[str], str | None]:
    """Build a citation -> full-document-text lookup over one or more BM25 indices.

    Each index is tried in order (pass laws first, then courts); the first exact
    citation match wins. Returns None when no index holds the citation.

    Args:
        *indices: ``BM25Index`` instances exposing ``get_by_citation``.

    Returns:
        ``lookup(citation) -> str | None``.
    """
    def lookup(citation: str) -> str | None:
        for index in indices:
            doc = index.get_by_citation(citation)
            if doc is not None:
                text = doc.get("text")
                if text:
                    return text
        return None

    return lookup


class Reranker:
    """Scores (query, document) pairs with a CrossEncoder, loading it on first use."""

    def __init__(self):
        self._model = None

    def rerank(
        self,
        query: str,
        citations: Iterable[str],
        text_lookup: Callable[[str], str | None],
    ) -> list[str]:
        """Keep only the citations the reranker is confident about.

        Each citation's full text is scored against the query; the raw logit is squashed
        to a [0,1] confidence (sigmoid) and citations clearing ``reranker_threshold`` are
        kept, best-first. If nothing clears the threshold, the top ``reranker_min_keep``
        by score are kept as a recall floor (a query should never silently return zero).

        Citations whose text cannot be found in any index are dropped (expected to be
        ~0, since the BM25 indices cover their full corpora).

        Args:
            query: The user's query.
            citations: Candidate citation strings (already fused/ranked).
            text_lookup: Maps a citation to its full document text (or None).

        Returns:
            The kept citation strings, ordered by descending confidence.
        """
        citations = list(citations)
        if not citations:
            return citations

        # Map citations to their document text; drop those with no recoverable text.
        scorable = [(c, t) for c in citations if (t := text_lookup(c))]
        min_keep = CONFIG.get("reranker_min_keep", 5)
        if not scorable:
            # Nothing to score -> keep the floor by the incoming (RRF) order.
            return citations[:min_keep]

        import torch

        if self._model is None:
            self._model = load_reranker()

        scores = self._model.predict(
            [(query, text) for _, text in scorable],
            batch_size=CONFIG.get("reranker_batch_size", 32),
            activation_fn=torch.nn.Sigmoid(),  # raw logits -> [0,1] confidence
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip((c for c, _ in scorable), (float(s) for s in scores)),
            key=lambda cs: cs[1],
            reverse=True,
        )

        threshold = CONFIG.get("reranker_threshold", 0.5)
        kept = [c for c, s in ranked if s >= threshold]
        if not kept:  # recall floor
            kept = [c for c, _ in ranked[:min_keep]]
        return kept


_RERANKER: Reranker | None = None


def get_reranker() -> Reranker:
    """Shared lazy singleton so the reranker model loads at most once per run."""
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = Reranker()
    return _RERANKER
