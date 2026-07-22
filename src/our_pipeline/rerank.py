from __future__ import annotations

import os

from dotenv import load_dotenv

# Lazily-loaded, process-wide local cross-encoder reranker (BGE-reranker-v2-m3).
_LOCAL_RERANKER = None


def _get_local_reranker(model_path: str):
    """Load (once) and cache the local cross-encoder reranker."""
    global _LOCAL_RERANKER
    if _LOCAL_RERANKER is None:
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading local reranker {model_path} on {device}…")
        _LOCAL_RERANKER = CrossEncoder(model_path, device=device, max_length=512)
    return _LOCAL_RERANKER


def local_rerank(
    query: str,
    candidates: list[dict],
    top_n: int,
    model_path: str,
    text_field: str = "text",
) -> list[dict]:
    """Rerank candidates with a local BGE cross-encoder (no API, fully offline).

    Mirrors ``cohere_rerank``'s contract so it is a drop-in replacement.

    Args:
        query: Search query string.
        candidates: Document dicts (must have ``text_field``).
        top_n: Number of top results to return.
        model_path: Local path to the cross-encoder (e.g. models/bge-reranker-v2-m3).
        text_field: Key for the document text scored against the query.

    Returns:
        top_n candidates sorted by relevance score, descending.
    """
    if not candidates:
        return []

    reranker = _get_local_reranker(model_path)
    pairs = [[query, doc.get(text_field, "")] for doc in candidates]
    scores = reranker.predict(pairs, batch_size=64, show_progress_bar=False)

    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    top_n = min(top_n, len(candidates))

    reranked = []
    for i in order[:top_n]:
        doc = candidates[i].copy()
        doc["_rerank_score"] = float(scores[i])
        reranked.append(doc)
    return reranked


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    citation_field: str = "citation",
) -> list[dict]:
    """Merge multiple ranked lists via Reciprocal Rank Fusion (k=60 from original paper).

    Args:
        ranked_lists: Each element is a ranked list of doc dicts (index 0 = rank 1).
        k: RRF smoothing constant.
        citation_field: Key used to uniquely identify documents.

    Returns:
        Merged list sorted by descending RRF score, with _rrf_score added.
    """
    scores: dict[str, float] = {}
    docs_by_citation: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            citation = doc.get(citation_field, "")
            if not citation:
                continue
            scores[citation] = scores.get(citation, 0.0) + 1.0 / (k + rank)
            if citation not in docs_by_citation:
                docs_by_citation[citation] = doc

    merged = []
    for citation, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        doc = docs_by_citation[citation].copy()
        doc["_rrf_score"] = score
        merged.append(doc)

    return merged


def cohere_rerank(
    query: str,
    candidates: list[dict],
    top_n: int,
    model: str = "rerank-multilingual-v3.0",
    api_key: str | None = None,
    text_field: str = "text",
) -> list[dict]:
    """Rerank candidates using Cohere Rerank API.

    Args:
        query: Search query string.
        candidates: List of document dicts (must have text_field).
        top_n: Number of top results to return.
        model: Cohere rerank model name.
        api_key: Cohere API key; falls back to COHERE_API_KEY env var.
        text_field: Key for the document text passed to the reranker.

    Returns:
        top_n candidates sorted by Cohere relevance score, descending.

    Raises:
        RuntimeError: If COHERE_API_KEY is missing.
        ImportError: If the cohere package is not installed.
    """
    import cohere

    load_dotenv()
    key = api_key or os.getenv("COHERE_API_KEY")
    if not key:
        raise RuntimeError("Missing COHERE_API_KEY. Add it to your .env file.")

    if not candidates:
        return []

    top_n = min(top_n, len(candidates))
    texts = [doc.get(text_field, "") for doc in candidates]

    co = cohere.ClientV2(api_key=key)
    response = co.rerank(
        query=query,
        documents=texts,
        model=model,
        top_n=top_n,
    )

    reranked = []
    for result in response.results:
        doc = candidates[result.index].copy()
        doc["_cohere_score"] = result.relevance_score
        reranked.append(doc)

    return reranked
