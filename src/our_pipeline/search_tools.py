from our_pipeline.constants import CONFIG
from our_pipeline.bm25.corpus import BM25Index
from our_pipeline.llm.keywords import extract_explicit_citations, extract_legal_keywords
from our_pipeline.llm.translate import translate_query
from our_pipeline.dense.index import DenseIndex


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    top_k: int | None = None,
    citation_field: str = "citation",
) -> list[dict]:
    """Fuse several ranked result lists with Reciprocal Rank Fusion.

    RRF combines results by rank rather than raw score, so it is robust to the
    incomparable BM25 score scales produced by query variants of different length
    or vocabulary. Each citation accrues ``1 / (k + rank)`` from every list it
    appears in (rank is 0-based). The document dict from the list where a citation
    ranked highest is kept for display and annotated with ``_rrf_score``.

    Args:
        ranked_lists: Per-variant result lists, each already sorted best-first.
        k: RRF constant (larger => flatter contribution across ranks).
        top_k: Trim the fused result to this many documents (None keeps all).
        citation_field: Dict key identifying a document.

    Returns:
        Fused documents sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    best_doc: dict[str, dict] = {}
    best_rank: dict[str, int] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            citation = doc.get(citation_field, "")
            if not citation:
                continue
            scores[citation] = scores.get(citation, 0.0) + 1.0 / (k + rank + 1)
            if citation not in best_rank or rank < best_rank[citation]:
                best_rank[citation] = rank
                best_doc[citation] = doc

    fused = sorted(best_doc.values(), key=lambda d: scores[d[citation_field]], reverse=True)
    for doc in fused:
        doc["_rrf_score"] = scores[doc[citation_field]]

    return fused[:top_k] if top_k is not None else fused


def retrieve_union(
    query: str,
    tools: dict,
    top_k: int | None = None,
    rrf_k: int = 60,
) -> list[str]:
    """Run every tool on the query and return their RRF-fused, deduped citations.

    Recall-first alternative to the ReAct agent: instead of letting the LLM curate a
    narrow ``Final Answer``, every registered tool is searched directly and their
    ranked hit lists are fused with Reciprocal Rank Fusion. Deterministic, and every
    tool — including both dense tools — is guaranteed to contribute.

    Args:
        query: The user's query (each tool applies its own expansion/translation/embedding).
        tools: Tool registry (name -> callable tool with ``get_last_citations``).
        top_k: Cap on returned citations after fusion (None keeps all).
        rrf_k: Reciprocal Rank Fusion constant.

    Returns:
        Fused citation strings, best-first, trimmed to ``top_k``.
    """
    ranked_lists: list[list[dict]] = []
    for tool in tools.values():
        try:
            tool(query)  # populates the tool's _last_results
            cites = tool.get_last_citations()
        except Exception:  # noqa: BLE001 - one failing tool (e.g. dense OOM) must not kill the query
            cites = []
        if cites:
            # Wrap as minimal docs; list order is the tool's rank order, which is all RRF needs.
            ranked_lists.append([{"citation": c} for c in cites])

    if not ranked_lists:
        return []

    fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k, top_k=top_k)
    return [doc["citation"] for doc in fused]


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
        self._expansion_cache: dict[str, tuple[str, list[str]]] = {}

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

        if CONFIG.get("law_query_expansion", True):
            self._last_results = self._expanded_search(query)
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

    def _expanded_search(self, query: str) -> list[dict]:
        """Search the German law corpus with extracted query variants, fused via RRF.

        The corpus is German-only, so instead of translating the whole (often verbose)
        question we extract concise German legal keywords and any statute articles the
        query names outright. Each variant is searched independently and the per-variant
        ranked lists are combined with Reciprocal Rank Fusion (scale-invariant, unlike
        the previous CombMAX over raw BM25 scores). Falls back to the raw query when
        extraction yields nothing.

        Args:
            query: Query string in any language.

        Returns:
            Fused list of document dicts, trimmed to ``self.top_k``.
        """
        if query not in self._expansion_cache:
            self._expansion_cache[query] = (
                extract_legal_keywords(query),
                extract_explicit_citations(query),
            )
        keywords, citations = self._expansion_cache[query]

        # 1. Guaranteed hits: articles the query names outright that exist in the corpus.
        #    BM25 ranks these unreliably (citation tokens carry little weight), so look
        #    them up directly instead of trusting the score.
        results: list[dict] = []
        seen: set[str] = set()
        for citation in citations:
            doc = self.index.get_by_citation(citation)
            if doc is not None and doc["citation"] not in seen:
                seen.add(doc["citation"])
                results.append(doc)

        # 2. Keyword BM25 search fills the remaining slots. RRF fuses multiple keyword
        #    variants when present (scale-invariant, unlike CombMAX over raw scores).
        variants = [keywords] if keywords else [query]  # fall back to raw query
        ranked_lists = [
            hits
            for v in variants
            if (hits := self.index.search(v, top_k=self.top_k, return_scores=True))
        ]
        if len(ranked_lists) == 1:
            keyword_hits = ranked_lists[0]
        elif ranked_lists:
            keyword_hits = reciprocal_rank_fusion(ranked_lists, k=CONFIG.get("rrf_k", 60))
        else:
            keyword_hits = []

        for doc in keyword_hits:
            if len(results) >= self.top_k:
                break
            if doc["citation"] not in seen:
                seen.add(doc["citation"])
                results.append(doc)

        return results[: self.top_k]

    def get_last_citations(self) -> list[str]:
        """Return citations from the last search.

        Returns:
            List of citation strings from the most recent search
        """
        return [doc.get("citation", "") for doc in self._last_results if doc.get("citation")]


class DenseSearchTool:
    """Tool for semantic (dense embedding) search over a legal corpus.

    One class serves both corpora: name/description are instance-level and the
    behavior is fixed by the DenseIndex it wraps. Embeddings are multilingual,
    so no query translation or keyword extraction is needed — natural-language
    queries match semantically related passages directly.
    """

    def __init__(
        self,
        index: DenseIndex,
        name: str,
        description: str,
        top_k: int = 5,
        max_excerpt_length: int = 300,
    ):
        """Initialize dense search tool.

        Args:
            index: DenseIndex for the corpus this instance searches
            name: Tool name the agent calls (e.g. "dense_search_laws")
            description: Tool description shown in the agent prompt
            top_k: Number of results to return
            max_excerpt_length: Maximum characters for text excerpts
        """
        self.index = index
        self.name = name
        self.description = description
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

        self._last_results = self.index.search(
            query,
            top_k=self.top_k,
            over_retrieve=CONFIG.get("dense_over_retrieve", 4),
        )

        if not self._last_results:
            return f"No relevant documents found for: '{query}'"

        formatted = []
        for doc in self._last_results:
            citation = doc.get("citation", "Unknown")
            text = doc.get("text", "")

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

