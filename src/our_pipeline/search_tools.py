from our_pipeline.constants import CONFIG
from our_pipeline.corpus import BM25Index
from our_pipeline.llm.translate import translate_query


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
        self._translation_cache: dict[str, dict[str, str]] = {}

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
        if query not in self._translation_cache:
            self._translation_cache[query] = translate_query(query)
        translations = self._translation_cache[query]

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
        self._translation_cache: dict[str, dict[str, str]] = {}

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
        if query not in self._translation_cache:
            self._translation_cache[query] = translate_query(query)
        translations = self._translation_cache[query]

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

