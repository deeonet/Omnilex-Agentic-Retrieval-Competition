import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

from constants import QUERY_FILE


class BM25Index:
    """BM25 index for keyword search over legal documents.

    Supports Swiss federal laws (SR) and court decisions (BGE).
    """

    def __init__(
        self,
        documents: list[dict] | None = None,
        text_field: str = "text",
        citation_field: str = "citation",
    ):
        """Initialize BM25 index.

        Args:
            documents: List of document dictionaries
            text_field: Key for document text in dict
            citation_field: Key for citation string in dict
        """
        self.text_field = text_field
        self.citation_field = citation_field

        self.documents: list[dict] = []
        self.index: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

        if documents:
            self.build(documents)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 indexing.

        Simple whitespace + lowercase tokenization.
        Can be overridden for language-specific tokenization.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase and split on non-alphanumeric characters
        text = text.lower()
        tokens = re.split(r"\W+", text)
        # Filter empty tokens
        return [t for t in tokens if t]

    def build(self, documents: list[dict]) -> None:
        """Build BM25 index from documents.

        Args:
            documents: List of document dictionaries
        """
        self.documents = documents

        # Tokenize all documents
        self._tokenized_corpus = []
        for doc in documents:
            text = doc.get(self.text_field, "")
            tokens = self.tokenize(text)
            self._tokenized_corpus.append(tokens)

        # Build BM25 index
        self.index = BM25Okapi(self._tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = False,
    ) -> list[dict]:
        """Search the index with a query.

        Args:
            query: Search query string
            top_k: Number of results to return
            return_scores: Whether to include BM25 scores in results

        Returns:
            List of matching documents (with optional scores)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")

        # Tokenize query
        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.index.get_scores(query_tokens)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue

            doc = self.documents[idx].copy()
            if return_scores:
                doc["_score"] = float(scores[idx])
            results.append(doc)

        return results

    def save(self, path: Path | str) -> None:
        """Save index to disk.

        Args:
            path: Path to save index (creates .pkl file)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "documents": self.documents,
            "tokenized_corpus": self._tokenized_corpus,
            "text_field": self.text_field,
            "citation_field": self.citation_field,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path | str) -> "BM25Index":
        """Load index from disk.

        Args:
            path: Path to saved index

        Returns:
            Loaded BM25Index instance
        """
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(
            text_field=data["text_field"],
            citation_field=data.get("citation_field", "citation"),
        )
        instance.documents = data["documents"]
        instance._tokenized_corpus = data["tokenized_corpus"]
        instance.index = BM25Okapi(instance._tokenized_corpus)

        return instance


def load_csv_corpus(
    csv_path: Path,
    chunk_size: int = 100_000,
    max_rows: int | None = None
) -> list[dict]:
    """Load CSV corpus into list of dicts with progress bar.
    
    Args:
        csv_path: Path to CSV file with 'citation' and 'text' columns
        chunk_size: Rows to process per chunk (for memory efficiency)
        max_rows: Optional limit on rows (for testing with smaller corpus)
    
    Returns:
        List of {"citation": str, "text": str} dicts
    """
    documents = []
    
    # Count rows for progress bar (fast line count)
    print(f"Counting rows in {csv_path.name}...")
    with open(csv_path, encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1  # minus header
    
    if max_rows:
        total_rows = min(total_rows, max_rows)
    print(f"Total rows to load: {total_rows:,}")
    
    rows_loaded = 0
    with tqdm(total=total_rows, desc=f"Loading {csv_path.name}") as pbar:
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                if max_rows and rows_loaded >= max_rows:
                    break
                documents.append({
                    "citation": str(row["citation"]),
                    "text": str(row["text"]) if pd.notna(row["text"]) else ""
                })
                rows_loaded += 1
            pbar.update(min(len(chunk), total_rows - pbar.n))
            if max_rows and rows_loaded >= max_rows:
                break
    
    return documents


def get_or_build_index(
    name: str,
    csv_path: Path,
    index_path: Path,
    force_rebuild: bool = False,
    max_rows: int | None = None
) -> BM25Index:
    """Load cached index or build from CSV.
    
    Args:
        name: Index name for logging
        csv_path: Path to corpus CSV
        index_path: Path to cache index pickle
        force_rebuild: If True, rebuild even if cache exists
        max_rows: Optional row limit (for testing with smaller corpus)
    
    Returns:
        BM25Index instance
    """
    # Use cached index if available and not forcing rebuild
    if index_path.exists() and not force_rebuild:
        print(f"Loading cached {name} index from {index_path}")
        try:
            index = BM25Index.load(index_path)
            print(f"  Loaded {len(index.documents):,} documents")
            return index
        except (EOFError, pickle.UnpicklingError, Exception) as e:
            print(f"  Warning: cached index is corrupted ({e}), rebuilding...")
            index_path.unlink(missing_ok=True)
    
    # Check CSV exists
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Creating empty index.")
        return BM25Index(documents=[])
    
    # Load corpus from CSV
    print(f"\n{'='*50}")
    print(f"Building {name} index from {csv_path}")
    print(f"{'='*50}")
    documents = load_csv_corpus(csv_path, max_rows=max_rows)
    
    if not documents:
        print(f"Warning: No documents loaded. Creating empty index.")
        return BM25Index(documents=[])
    
    # Build BM25 index
    print(f"\nBuilding BM25 index for {len(documents):,} documents...")
    index = BM25Index(
        documents=documents,
        text_field="text",
        citation_field="citation"
    )
    print(f"Index built successfully!")
    
    # Cache index for future runs
    print(f"Saving index to {index_path}...")
    index.save(index_path)
    print(f"Index cached.")
    
    return index


def get_query_file():
    # Load queries from the configured query file
    query_file = QUERY_FILE
    if not query_file.exists():
        raw_query_file = QUERY_FILE.parent / "raw" / QUERY_FILE.name
        if raw_query_file.exists():
            query_file = raw_query_file
        else:
            raise FileNotFoundError(f"Query file not found: {QUERY_FILE}")
    return query_file


# ---------------------------------------------------------------------------
# Dense (semantic) index backed by FAISS + Mistral embeddings
# ---------------------------------------------------------------------------

_DEFAULT_EMBED_API_BASE = "https://chat-ai.academiccloud.de/v1"
_DEFAULT_EMBED_MODEL = "e5-mistral-7b-instruct"


class DenseIndex:
    """Semantic vector index using FAISS + local sentence-transformers or remote API.

    When ``model`` is a local directory path that exists on disk, embeddings are
    produced locally via ``sentence_transformers.SentenceTransformer`` (GPU if
    available).  Otherwise the OpenAI-compatible API at ``api_base`` is used.

    The FAISS IndexFlatIP provides exact inner-product (cosine) search.
    """

    def __init__(
        self,
        documents: list[dict] | None = None,
        text_field: str = "text",
        citation_field: str = "citation",
        model: str = _DEFAULT_EMBED_MODEL,
        api_key: str | None = None,
        api_base: str = _DEFAULT_EMBED_API_BASE,
        batch_size: int = 32,
    ):
        load_dotenv()
        self.text_field = text_field
        self.citation_field = citation_field
        self.model = model
        self.api_key = api_key or os.getenv("API_KEY")
        self.api_base = api_base
        self.batch_size = batch_size

        self.documents: list[dict] = []
        self._faiss_index = None
        self._embeddings: np.ndarray | None = None
        self._st_model = None  # lazy-loaded sentence-transformer

        if documents:
            self.build(documents)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _is_local_model(self) -> bool:
        return Path(self.model).exists()

    def _embed_texts(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Embed texts locally or via API, returning a float32 matrix."""
        if self._is_local_model():
            return self._embed_local(texts, is_query=is_query)
        return self._embed_api(texts)

    def _embed_local(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Embed using a local sentence-transformers model.

        Uses all available CUDA GPUs via encode_multi_process for corpus
        embedding (is_query=False).  Single-GPU or CPU for query embedding.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        primary_device = "cuda:0" if n_gpus > 0 else "cpu"

        if self._st_model is None:
            print(f"Loading local embedding model from {self.model} on {primary_device}…")
            print(f"  Detected {n_gpus} CUDA GPU(s)")
            self._st_model = SentenceTransformer(self.model, device=primary_device)

        prefix = "query: " if is_query else "passage: "
        prefixed = [prefix + (t if t.strip() else " ") for t in texts]

        if n_gpus > 1 and not is_query:
            devices = [f"cuda:{i}" for i in range(n_gpus)]
            print(f"  Spreading corpus embedding across {devices}")
            pool = self._st_model.start_multi_process_pool(target_devices=devices)
            try:
                embeddings = self._st_model.encode_multi_process(
                    prefixed, pool, batch_size=self.batch_size
                )
            finally:
                self._st_model.stop_multi_process_pool(pool)
        else:
            embeddings = self._st_model.encode(
                prefixed,
                normalize_embeddings=True,
                batch_size=self.batch_size,
                show_progress_bar=True,
            )
        return np.array(embeddings, dtype=np.float32)

    def _embed_api(self, texts: list[str]) -> np.ndarray:
        """Call the remote OpenAI-compatible embeddings endpoint in batches."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        all_embeddings: list[list[float]] = []

        with tqdm(total=len(texts), desc="Embedding documents", unit="doc") as pbar:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch = [t if t.strip() else " " for t in batch]
                response = client.embeddings.create(model=self.model, input=batch)
                all_embeddings.extend(e.embedding for e in response.data)
                pbar.update(len(batch))

        return np.array(all_embeddings, dtype=np.float32)

    # ------------------------------------------------------------------
    # Build / Search
    # ------------------------------------------------------------------

    def build(self, documents: list[dict]) -> None:
        """Embed all documents and build a FAISS IndexFlatIP."""
        import faiss

        self.documents = documents
        texts = [doc.get(self.text_field, "") for doc in documents]

        print(f"Building dense index: embedding {len(texts):,} documents with {self.model}…")
        self._embeddings = self._embed_texts(texts)

        # Normalise to unit length → inner product == cosine similarity
        faiss.normalize_L2(self._embeddings)

        dim = self._embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(self._embeddings)
        print(f"Dense index ready: {self._faiss_index.ntotal:,} vectors, dim={dim}")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return top-k documents by cosine similarity to the query."""
        import faiss

        if self._faiss_index is None:
            raise ValueError("Dense index not built. Call build() first.")

        query_emb = self._embed_texts([query], is_query=True)
        if not self._is_local_model():
            faiss.normalize_L2(query_emb)  # local model already normalizes

        scores, indices = self._faiss_index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx].copy()
            doc["_score"] = float(score)
            results.append(doc)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir_path: Path | str) -> None:
        """Persist FAISS index + documents to a directory."""
        import faiss

        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._faiss_index, str(dir_path / "index.faiss"))
        np.save(dir_path / "embeddings.npy", self._embeddings)

        meta = {
            "documents": self.documents,
            "text_field": self.text_field,
            "citation_field": self.citation_field,
            "model": self.model,
            "api_base": self.api_base,
        }
        with open(dir_path / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(
        cls,
        dir_path: Path | str,
        api_key: str | None = None,
    ) -> "DenseIndex":
        """Load a previously saved DenseIndex from a directory."""
        import faiss

        dir_path = Path(dir_path)

        with open(dir_path / "meta.pkl", "rb") as f:
            meta = pickle.load(f)

        instance = cls(
            text_field=meta["text_field"],
            citation_field=meta.get("citation_field", "citation"),
            model=meta.get("model", _DEFAULT_EMBED_MODEL),
            api_key=api_key,
            api_base=meta.get("api_base", _DEFAULT_EMBED_API_BASE),
        )
        instance.documents = meta["documents"]
        instance._embeddings = np.load(dir_path / "embeddings.npy")
        instance._faiss_index = faiss.read_index(str(dir_path / "index.faiss"))

        return instance


def get_or_build_dense_index(
    name: str,
    csv_path: Path,
    index_dir: Path,
    force_rebuild: bool = False,
    max_rows: int | None = None,
    model: str = _DEFAULT_EMBED_MODEL,
    api_key: str | None = None,
    api_base: str = _DEFAULT_EMBED_API_BASE,
    batch_size: int = 32,
) -> DenseIndex:
    """Load a cached DenseIndex or build one from a CSV corpus.

    Args:
        name: Human-readable name for logging.
        csv_path: Path to corpus CSV with 'citation' and 'text' columns.
        index_dir: Directory for the FAISS index artefacts.
        force_rebuild: Rebuild even when a valid cache exists.
        max_rows: Limit the corpus size (useful for the large courts corpus).
        model: Embedding model name.
        api_key: API key for the embedding service.
        api_base: Base URL for the embedding service.
        batch_size: Documents per embedding API call.

    Returns:
        Loaded or freshly built DenseIndex.
    """
    meta_path = index_dir / "meta.pkl"

    if meta_path.exists() and not force_rebuild:
        print(f"Loading cached dense {name} index from {index_dir}")
        try:
            idx = DenseIndex.load(index_dir, api_key=api_key)
            print(f"  Loaded {len(idx.documents):,} documents")
            return idx
        except Exception as e:
            print(f"  Warning: cached dense index corrupted ({e}), rebuilding…")

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Creating empty dense index.")
        return DenseIndex(documents=[])

    print(f"\n{'='*50}")
    print(f"Building dense {name} index from {csv_path}")
    if max_rows:
        print(f"  (limited to {max_rows:,} rows)")
    print(f"{'='*50}")

    documents = load_csv_corpus(csv_path, max_rows=max_rows)

    if not documents:
        print("Warning: No documents loaded. Creating empty dense index.")
        return DenseIndex(documents=[])

    idx = DenseIndex(
        documents=documents,
        model=model,
        api_key=api_key,
        api_base=api_base,
        batch_size=batch_size,
    )

    print(f"Saving dense {name} index to {index_dir}…")
    idx.save(index_dir)
    print("Dense index cached.")

    return idx