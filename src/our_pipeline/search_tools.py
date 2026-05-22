from our_pipeline.corpus import BM25Index

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

        results = self.index.search(query, top_k=self.top_k)
        self._last_results = results

        if not results:
            return f"No relevant federal laws found for: '{query}'"

        formatted = []
        for doc in results:
            citation = doc.get("citation", "Unknown")
            text = doc.get("text", "")

            # Truncate text for readability
            if len(text) > self.max_excerpt_length:
                text = text[: self.max_excerpt_length] + "..."

            formatted.append(f"- {citation}: {text}")

        return "\n".join(formatted)

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

        results = self.index.search(query, top_k=self.top_k)
        self._last_results = results

        if not results:
            return f"No relevant court decisions found for: '{query}'"

        formatted = []
        for doc in results:
            citation = doc.get("citation", "Unknown")
            text = doc.get("text", "")

            # Truncate text for readability
            if len(text) > self.max_excerpt_length:
                text = text[: self.max_excerpt_length] + "..."

            formatted.append(f"- {citation}: {text}")

        return "\n".join(formatted)

    def get_last_citations(self) -> list[str]:
        """Return citations from the last search.

        Returns:
            List of citation strings from the most recent search
        """
        return [doc.get("citation", "") for doc in self._last_results if doc.get("citation")]

