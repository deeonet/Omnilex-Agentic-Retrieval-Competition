import pickle
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm.notebook import tqdm

from our_pipeline.constants import CONFIG, QUERY_FILE

# Curated German + legal-filler stopwords. Kept inline so tokenization needs no
# external corpus download (nltk/spacy are not installed in this environment).
_STOPWORDS: frozenset[str] = frozenset(
    """
    aber alle allem allen aller alles als also am an ander andere anderem anderen
    anderer anderes anderm andern anderr anders auch auf aus bei bin bis bist da
    damit dann der den des dem die das dass daß derselbe derselben denselben
    desselben demselben dieselbe dieselben dasselbe dazu dein deine deinem deinen
    deiner deines denn derer dessen dich dir du dies diese diesem diesen dieser
    dieses doch dort durch ein eine einem einen einer eines einig einige einigem
    einigen einiger einiges einmal er ihn ihm es etwas euer eure eurem euren eurer
    eures für gegen gewesen hab habe haben hat hatte hatten hier hin hinter ich
    mich mir ihr ihre ihrem ihren ihrer ihres euch im in indem ins ist jede jedem
    jeden jeder jedes jene jenem jenen jener jenes jetzt kann kein keine keinem
    keinen keiner keines können könnte machen man manche manchem manchen mancher
    manches mein meine meinem meinen meiner meines mit muss musste nach nicht
    nichts noch nun nur ob oder ohne sehr sein seine seinem seinen seiner seines
    selbst sich sie ihnen sind so solche solchem solchen solcher solches soll
    sollte sondern sonst über um und uns unse unsem unsen unser unses unter viel
    vom von vor während war waren warst was weg weil weiter welche welchem welchen
    welcher welches wenn werde werden wie wieder will wir wird wirst wo wollen
    wollte würde würden zu zum zur zwar zwischen
    bzw sowie gemäss gemäß dabei jedoch sowohl insbesondere bzgl ggf etc
    """.split()
)

try:  # Snowball German stemmer is optional; tokenization degrades gracefully without it.
    import snowballstemmer

    _STEMMER = snowballstemmer.stemmer("german")
except Exception:  # pragma: no cover - exercised only when the package is missing
    _STEMMER = None


@lru_cache(maxsize=500_000)
def _stem(token: str) -> str:
    """Stem a single token via Snowball (cached); identity if stemmer unavailable."""
    return _STEMMER.stemWord(token) if _STEMMER is not None else token


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
        self._citation_to_doc: dict[str, dict] = {}

        if documents:
            self.build(documents)

    @staticmethod
    def _citation_key(citation: str) -> str:
        """Whitespace-normalized key for exact citation lookup."""
        return re.sub(r"\s+", " ", citation).strip()

    def _build_citation_map(self) -> None:
        """Index documents by exact citation for direct (non-BM25) lookup."""
        self._citation_to_doc = {
            self._citation_key(doc.get(self.citation_field, "")): doc
            for doc in self.documents
            if doc.get(self.citation_field)
        }

    def get_by_citation(self, citation: str) -> dict | None:
        """Return the document whose citation exactly matches ``citation`` (or None).

        Used to guarantee retrieval of articles a query names outright, which BM25
        ranks unreliably because citation tokens have little discriminative weight.
        """
        return self._citation_to_doc.get(self._citation_key(citation))

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 indexing.

        Lowercase + non-word split, then optionally drop German/legal stopwords and
        apply Snowball German stemming (both gated by CONFIG). The exact same path is
        used for corpus documents and queries, so they stay consistent by construction.

        NOTE: changing the stopword/stemming flags requires rebuilding the index.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        text = text.lower()
        tokens = [t for t in re.split(r"\W+", text) if t]

        if CONFIG.get("bm25_remove_stopwords", True):
            tokens = [t for t in tokens if t not in _STOPWORDS]

        if CONFIG.get("bm25_use_stemming", True):
            tokens = [_stem(t) for t in tokens]

        return tokens

    def _searchable_text(self, doc: dict) -> str:
        """Compose the text that gets indexed for a document.

        Includes the citation and title (when present) alongside the body so that
        explicit article references and law-code tokens in a query can match the
        right provision. ``doc[text_field]`` itself is left untouched for display.
        """
        parts = [
            doc.get(self.citation_field, ""),
            doc.get("title", ""),
            doc.get(self.text_field, ""),
        ]
        return " ".join(p for p in parts if p)

    def build(self, documents: list[dict]) -> None:
        """Build BM25 index from documents.

        Args:
            documents: List of document dictionaries
        """
        self.documents = documents

        # Tokenize the searchable blob (citation + title + text) for each document
        self._tokenized_corpus = [self.tokenize(self._searchable_text(doc)) for doc in documents]

        # Build BM25 index with tunable parameters
        self.index = BM25Okapi(
            self._tokenized_corpus,
            k1=CONFIG.get("bm25_k1", 1.5),
            b=CONFIG.get("bm25_b", 0.75),
        )
        self._build_citation_map()

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
        instance.index = BM25Okapi(
            instance._tokenized_corpus,
            k1=CONFIG.get("bm25_k1", 1.5),
            b=CONFIG.get("bm25_b", 0.75),
        )
        instance._build_citation_map()

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
                doc = {
                    "citation": str(row["citation"]),
                    "text": str(row["text"]) if pd.notna(row["text"]) else "",
                }
                # Laws have a title column; court decisions do not.
                if "title" in chunk.columns and pd.notna(row["title"]):
                    doc["title"] = str(row["title"])
                documents.append(doc)
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