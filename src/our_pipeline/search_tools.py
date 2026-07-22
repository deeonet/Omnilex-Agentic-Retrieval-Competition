import os

from dotenv import load_dotenv

from constants import CONFIG
from corpus import BM25Index
from llm.translate import translate_query


class LawSearchTool:
    """Tool for searching Swiss federal laws corpus.

    Searches the SR (Systematische Rechtssammlung) collection
    using BM25 keyword matching.
    """

    name: str = "search_laws"
    description: str = """Search Swiss federal laws (SR/Systematische Rechtssammlung) by keywords.
Input: Search query string (can be in German, French, Italian, or English)
Output: List of relevant law citations with text excerpts

Use this tool to find relevant federal law provisions for a legal question.
Example queries: "contract formation requirements", "Vertragsabschluss", "divorce grounds"
"""

    def __init__(
        self,
        index: BM25Index,
        top_k: int = 5,
        max_excerpt_length: int = 300,
    ):
        """Initialize law search tool.

        Args:
            index: BM25Index for federal laws corpus
            top_k: Number of results to return
            max_excerpt_length: Maximum characters for text excerpts
        """
        self.index = index
        self.top_k = top_k
        self.max_excerpt_length = max_excerpt_length
        self._last_results: list[dict] = []

    def __call__(self, query: str) -> str:
        """Execute search and return formatted results.

        Args:
            query: Search query string

        Returns:
            Formatted string with search results
        """
        return self.run(query)

    def run(self, query: str) -> str:
        """Execute search and return formatted results.

        Args:
            query: Search query string

        Returns:
            Formatted string with search results
        """
        if not query or not query.strip():
            self._last_results = []
            return "Error: Empty query. Please provide search terms."

        if CONFIG.get("enable_multilingual_search", False):
            self._last_results = self._multilingual_search(query)
        else:
            self._last_results = self.index.search(query, top_k=self.top_k)

        if not self._last_results:
            return f"No relevant federal laws found for: '{query}'"

        formatted = []
        for doc in self._last_results:
            citation = doc.get("citation", "Unknown")
            text = doc.get("text", "")

            if len(text) > self.max_excerpt_length:
                text = text[: self.max_excerpt_length] + "..."

            formatted.append(f"- {citation}: {text}")

        return "\n".join(formatted)

    def _multilingual_search(self, query: str) -> list[dict]:
        """Search with EN/DE/FR/IT query variants and fuse results via CombMAX.

        Translates the query to English, German, French, and Italian, runs a BM25
        search for each unique variant, and keeps the highest-scoring
        document per citation across all language results.

        Args:
            query: Query string in any language.

        Returns:
            List of document dicts with ``_score`` key, sorted descending,
            trimmed to ``self.top_k``.
        """
        translations = translate_query(query)

        # Collect unique query variants; fall back to original if translation failed
        seen: set[str] = set()
        queries: list[str] = []
        for q in [translations.get("en"), translations.get("de"), translations.get("fr"), translations.get("it"), query]:
            if q and q.strip() and q not in seen:
                seen.add(q)
                queries.append(q)

        best_by_citation: dict[str, dict] = {}
        for q in queries:
            for doc in self.index.search(q, top_k=self.top_k, return_scores=True):
                citation = doc.get("citation", "")
                if not citation:
                    continue
                existing = best_by_citation.get(citation)
                if existing is None or doc["_score"] > existing["_score"]:
                    best_by_citation[citation] = doc

        merged = sorted(best_by_citation.values(), key=lambda d: d["_score"], reverse=True)
        return merged[: self.top_k]

    def get_last_citations(self) -> list[str]:
        """Return citations from the last search.

        Returns:
            List of citation strings from the most recent search
        """
        return [doc.get("citation", "") for doc in self._last_results if doc.get("citation")]

    def get_last_results(self) -> list[dict]:
        """Return the full result docs (citation + text + score) from the last search."""
        return self._last_results


class CourtSearchTool:
    """Tool for searching Swiss Federal Court decisions corpus.

    Searches court decisions (BGE and docket-style citations)
    using BM25 keyword matching.
    """

    name: str = "search_courts"
    description: str = """Search Swiss Federal Court decisions by keywords.
Input: Search query string (German, French, Italian, or English)
Output: List of relevant court decision citations with excerpts

Use this tool to find relevant case law and judicial interpretations.
Example queries: "negligence standard of care", "Sorgfaltspflicht", "contract interpretation"
"""

    def __init__(
        self,
        index: BM25Index,
        top_k: int = 5,
        max_excerpt_length: int = 300,
    ):
        """Initialize court search tool.

        Args:
            index: BM25Index for court decisions corpus
            top_k: Number of results to return
            max_excerpt_length: Maximum characters for text excerpts
        """
        self.index = index
        self.top_k = top_k
        self.max_excerpt_length = max_excerpt_length
        self._last_results: list[dict] = []

    def __call__(self, query: str) -> str:
        """Execute search and return formatted results.

        Args:
            query: Search query string

        Returns:
            Formatted string with search results
        """
        return self.run(query)

    def run(self, query: str) -> str:
        """Execute search and return formatted results.

        Args:
            query: Search query string

        Returns:
            Formatted string with search results
        """
        if not query or not query.strip():
            self._last_results = []
            return "Error: Empty query. Please provide search terms."

        if CONFIG.get("enable_multilingual_search", False):
            self._last_results = self._multilingual_search(query)
        else:
            self._last_results = self.index.search(query, top_k=self.top_k)

        if not self._last_results:
            return f"No relevant court decisions found for: '{query}'"

        formatted = []
        for doc in self._last_results:
            citation = doc.get("citation", "Unknown")
            text = doc.get("text", "")

            if len(text) > self.max_excerpt_length:
                text = text[: self.max_excerpt_length] + "..."

            formatted.append(f"- {citation}: {text}")

        return "\n".join(formatted)

    def _multilingual_search(self, query: str) -> list[dict]:
        """Search with EN/DE/FR/IT query variants and fuse results via CombMAX.

        Translates the query to English, German, French, and Italian, runs a BM25
        search for each unique variant, and keeps the highest-scoring
        document per citation across all language results.

        Args:
            query: Query string in any language.

        Returns:
            List of document dicts with ``_score`` key, sorted descending,
            trimmed to ``self.top_k``.
        """
        translations = translate_query(query)

        seen: set[str] = set()
        queries: list[str] = []
        for q in [translations.get("en"), translations.get("de"), translations.get("fr"), translations.get("it"), query]:
            if q and q.strip() and q not in seen:
                seen.add(q)
                queries.append(q)

        best_by_citation: dict[str, dict] = {}
        for q in queries:
            for doc in self.index.search(q, top_k=self.top_k, return_scores=True):
                citation = doc.get("citation", "")
                if not citation:
                    continue
                existing = best_by_citation.get(citation)
                if existing is None or doc["_score"] > existing["_score"]:
                    best_by_citation[citation] = doc

        merged = sorted(best_by_citation.values(), key=lambda d: d["_score"], reverse=True)
        return merged[: self.top_k]

    def get_last_citations(self) -> list[str]:
        """Return citations from the last search.

        Returns:
            List of citation strings from the most recent search
        """
        return [doc.get("citation", "") for doc in self._last_results if doc.get("citation")]

    def get_last_results(self) -> list[dict]:
        """Return the full result docs (citation + text + score) from the last search."""
        return self._last_results


# ---------------------------------------------------------------------------
# Hybrid BM25 + Dense retrieval with RRF fusion and Cohere reranking
# ---------------------------------------------------------------------------

class HybridSearchTool:
    """BM25 + dense retrieval fused via RRF, then reranked by Cohere.

    Pipeline per query:
      1. Multilingual BM25 (CombMAX across EN/DE/FR/IT) → top-N candidates
      2. Dense search via DenseIndex (mistral-embed)     → top-N candidates
      3. Reciprocal Rank Fusion (k=60)                  → merged top-M list
      4. Cohere Rerank                                   → final top-K results

    Falls back gracefully when the dense index or Cohere key is absent.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        dense_index,  # DenseIndex | None
        corpus_type: str,  # "laws" or "courts"
        top_k: int = 10,
        max_excerpt_length: int = 300,
    ):
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.corpus_type = corpus_type
        self.top_k = top_k
        self.max_excerpt_length = max_excerpt_length
        self._last_results: list[dict] = []

        if corpus_type == "laws":
            self.name = "search_laws"
            self.description = LawSearchTool.description
        else:
            self.name = "search_courts"
            self.description = CourtSearchTool.description

    def __call__(self, query: str) -> str:
        return self.run(query)

    def run(self, query: str) -> str:
        if not query or not query.strip():
            self._last_results = []
            return "Error: Empty query. Please provide search terms."

        candidate_k = CONFIG.get("dense_candidate_k", 50)

        # --- Stage 1: BM25 ---
        if CONFIG.get("enable_multilingual_search", False):
            bm25_results = self._multilingual_bm25(query, candidate_k)
        else:
            bm25_results = self.bm25_index.search(query, top_k=candidate_k, return_scores=True)

        # --- Stage 2: Dense ---
        dense_results: list[dict] = []
        if self.dense_index is not None:
            try:
                dense_results = self.dense_index.search(query, top_k=candidate_k)
            except Exception as e:
                print(f"Dense search failed: {e}. Using BM25 only.")

        # --- Stage 3: RRF fusion ---
        from rerank import reciprocal_rank_fusion, cohere_rerank, local_rerank

        if dense_results:
            candidates = reciprocal_rank_fusion(
                [bm25_results, dense_results],
                k=CONFIG.get("rrf_k", 60),
            )
        else:
            candidates = bm25_results

        rerank_n = CONFIG.get("cohere_rerank_candidates", 50)
        candidates = candidates[:rerank_n]

        # --- Stage 4: rerank (local BGE cross-encoder / Cohere API / none) ---
        backend = CONFIG.get("rerank_backend", "local")
        if backend == "local" and candidates:
            try:
                candidates = local_rerank(
                    query=query,
                    candidates=candidates,
                    top_n=self.top_k,
                    model_path=CONFIG.get("local_rerank_model", "models/bge-reranker-v2-m3"),
                )
            except Exception as e:
                print(f"Local rerank failed: {e}. Using RRF results.")
                candidates = candidates[: self.top_k]
        elif backend == "cohere" and candidates:
            load_dotenv()
            cohere_key = os.getenv("COHERE_API_KEY")
            if cohere_key:
                try:
                    candidates = cohere_rerank(
                        query=query,
                        candidates=candidates,
                        top_n=self.top_k,
                        model=CONFIG.get("cohere_rerank_model", "rerank-multilingual-v3.0"),
                        api_key=cohere_key,
                    )
                except Exception as e:
                    print(f"Cohere rerank failed: {e}. Using RRF results.")
                    candidates = candidates[: self.top_k]
            else:
                candidates = candidates[: self.top_k]
        else:
            candidates = candidates[: self.top_k]

        self._last_results = candidates

        if not self._last_results:
            label = "federal laws" if self.corpus_type == "laws" else "court decisions"
            return f"No relevant {label} found for: '{query}'"

        formatted = []
        for doc in self._last_results:
            citation = doc.get("citation", "Unknown")
            text = doc.get("text", "")
            if len(text) > self.max_excerpt_length:
                text = text[: self.max_excerpt_length] + "..."
            formatted.append(f"- {citation}: {text}")

        return "\n".join(formatted)

    def _multilingual_bm25(self, query: str, candidate_k: int) -> list[dict]:
        """CombMAX BM25 fusion across EN/DE/FR/IT query variants."""
        translations = translate_query(query)

        seen: set[str] = set()
        queries: list[str] = []
        for q in [
            translations.get("en"),
            translations.get("de"),
            translations.get("fr"),
            translations.get("it"),
            query,
        ]:
            if q and q.strip() and q not in seen:
                seen.add(q)
                queries.append(q)

        best_by_citation: dict[str, dict] = {}
        for q in queries:
            for doc in self.bm25_index.search(q, top_k=candidate_k, return_scores=True):
                citation = doc.get("citation", "")
                if not citation:
                    continue
                existing = best_by_citation.get(citation)
                if existing is None or doc["_score"] > existing["_score"]:
                    best_by_citation[citation] = doc

        merged = sorted(best_by_citation.values(), key=lambda d: d["_score"], reverse=True)
        return merged[:candidate_k]

    def get_last_citations(self) -> list[str]:
        return [doc.get("citation", "") for doc in self._last_results if doc.get("citation")]

    def get_last_results(self) -> list[dict]:
        """Return the full result docs (citation + text + score) from the last search."""
        return self._last_results
