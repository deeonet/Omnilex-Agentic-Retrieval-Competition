import pandas as pd
from tqdm.notebook import tqdm
import pickle
import re
from rank_bm25 import BM25Okapi
from pathlib import Path
from our_pipeline.constants import QUERY_FILE


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
        index = BM25Index.load(index_path)
        print(f"  Loaded {len(index.documents):,} documents")
        return index
    
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