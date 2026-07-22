"""FAISS-backed dense index over a legal corpus.

Wraps a FAISS index (cosine similarity via normalized inner product) plus a
positionally aligned metadata table (citation/excerpt/title) — both produced
offline by ``our_pipeline.dense.build_embeddings``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from our_pipeline.dense.embedder import get_query_embedder


class DenseIndex:
    """Semantic search over precomputed corpus embeddings."""

    def __init__(self, index, meta: pd.DataFrame):
        """Use :meth:`load` instead of constructing directly.

        Args:
            index: FAISS index whose id i corresponds to ``meta`` row i.
            meta: DataFrame with columns ``citation``, ``excerpt`` (+ ``title``
                for the laws corpus), in corpus CSV row order.
        """
        self.index = index
        self.meta = meta

    @classmethod
    def load(cls, faiss_path: Path | str, meta_path: Path | str) -> "DenseIndex":
        """Load the FAISS index and metadata, validating their alignment."""
        import faiss

        index = faiss.read_index(str(faiss_path))
        meta = pd.read_parquet(meta_path)
        if index.ntotal != len(meta):
            raise ValueError(
                f"Dense index/metadata mismatch: {faiss_path} has {index.ntotal} vectors "
                f"but {meta_path} has {len(meta)} rows. Rebuild with build_embeddings."
            )
        return cls(index, meta)

    def search(
        self,
        query: str | np.ndarray,
        top_k: int = 40,
        over_retrieve: int = 4,
    ) -> list[dict]:
        """Return the ``top_k`` most similar documents, deduplicated by citation.

        Citations repeat across rows (paragraph granularity, especially for court
        considerations), so ``top_k * over_retrieve`` vectors are fetched and the
        best-scoring row per citation is kept.

        Args:
            query: Query string (embedded via the shared QueryEmbedder; empty
                strings yield []) or a precomputed normalized embedding.
            top_k: Number of documents to return.
            over_retrieve: Over-retrieval factor compensating for duplicates.

        Returns:
            Documents as ``{"citation", "text", "title"?, "_score"}`` dicts —
            the same schema BM25 results use, sorted by descending similarity.
        """
        if isinstance(query, str):
            if not query.strip():
                return []
            vec = get_query_embedder().embed(query)
        else:
            vec = np.asarray(query, dtype=np.float32)

        n_fetch = min(max(top_k * over_retrieve, top_k), 512, self.index.ntotal)
        scores, ids = self.index.search(vec.reshape(1, -1).astype(np.float32), n_fetch)

        has_title = "title" in self.meta.columns
        results: list[dict] = []
        seen: set[str] = set()
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:  # FAISS pads with -1 when fewer than n_fetch hits exist
                continue
            row = self.meta.iloc[int(idx)]
            citation = row["citation"]
            if not citation or citation in seen:
                continue
            seen.add(citation)
            doc = {"citation": citation, "text": row["excerpt"], "_score": float(score)}
            if has_title and row["title"]:
                doc["title"] = row["title"]
            results.append(doc)
            if len(results) >= top_k:
                break

        return results
