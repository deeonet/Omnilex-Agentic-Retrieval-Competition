"""LightRAG-backed search tool for legal citation retrieval.

Mirrors the search-tool contract used by the ReAct agent (see
``src/our_pipeline/search_tools.py`` and ``GraphRAGLocalSearchTool``):
``name``, ``description``, ``__call__(query) -> str`` returning
``"- {citation}: ..."`` lines, plus ``get_last_citations() -> list[str]``.

LightRAG is used purely as a retriever: we request the retrieved *context*
(``only_need_context=True``) — no LLM-generated answer — and recover the source
citations. Because every indexed document was inserted with its citation
prefixed as ``[citation]`` (see scripts/build_lightrag_index.py), the citation
strings appear verbatim in retrieved chunk text. We keep only those that exist
in the corpus, guaranteeing closed-vocabulary output.
"""

from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_LAWS_WORKDIR = PROJECT_ROOT / "lightrag" / "laws"
DEFAULT_LAWS_CSV = PROJECT_ROOT / "data" / "laws_de.csv"

# Matches the ``[citation]`` markers we prefix onto every indexed document.
_BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,120})\]")


class LightRAGSearchTool:
    """Graph-augmented retrieval over the Swiss legal knowledge graph (LightRAG)."""

    name: str = "search_laws_graph"
    description: str = """Search Swiss federal laws using a graph-augmented retriever (LightRAG).
Input: A specific legal question, concept, or provision to look up.
Output: List of relevant law citations.

Use this in addition to keyword search when a question involves related or
connected provisions — the knowledge graph links articles that share legal
concepts, principles, and parties.
Example queries: "formation of a contract", "liability for damages", "Art. 41 OR"
"""

    def __init__(
        self,
        workdir: Path | str = DEFAULT_LAWS_WORKDIR,
        citation_csv: Path | str = DEFAULT_LAWS_CSV,
        mode: str = "mix",
        top_k: int = 40,
        chunk_top_k: int = 20,
    ):
        self.workdir = Path(workdir)
        self.mode = mode
        self.top_k = top_k
        self.chunk_top_k = chunk_top_k
        self._citation_set = self._load_citation_set(Path(citation_csv))
        self._rag = None
        # LightRAG spawns persistent background worker tasks, so it needs an
        # event loop that stays alive across calls. We run one in a daemon
        # thread and submit coroutines to it (run_agent calls this tool
        # synchronously, possibly many times per query).
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._last_citations: list[str] = []
        self._last_results: list[dict] = []
        self._last_context: str = ""

    def _run(self, coro):
        """Run a coroutine on the tool's persistent background loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    @staticmethod
    def _load_citation_set(csv_path: Path) -> set[str]:
        df = pd.read_csv(csv_path, usecols=["citation"])
        return {str(c) for c in df["citation"].dropna().unique()}

    def _ensure_loaded(self) -> None:
        if self._rag is None:
            from omnilex.retrieval.lightrag_config import make_lightrag

            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._thread.start()
            self._rag = self._run(make_lightrag(self.workdir))

    def __call__(self, query: str) -> str:
        return self.run(query)

    def run(self, query: str) -> str:
        if not query or not query.strip():
            return "Error: Empty query. Please provide search terms."
        self._ensure_loaded()
        from lightrag import QueryParam

        param = QueryParam(
            mode=self.mode,
            only_need_context=True,
            top_k=self.top_k,
            chunk_top_k=self.chunk_top_k,
            enable_rerank=False,
        )
        context = self._run(self._rag.aquery(query, param=param))
        self._last_context = context if isinstance(context, str) else str(context)
        self._last_citations = self._recover_citations(self._last_context)
        # run_agent consumes get_last_results(); ranked by appearance order.
        self._last_results = [
            {"citation": c, "text": "", "_score": None} for c in self._last_citations
        ]

        if not self._last_citations:
            return f"No relevant federal laws found for: '{query}'"
        return "\n".join(f"- {c}" for c in self._last_citations)

    def _recover_citations(self, context: str) -> list[str]:
        """Keep only ``[citation]`` markers that exist verbatim in the corpus."""
        found: list[str] = []
        seen: set[str] = set()
        for raw in _BRACKET_RE.findall(context):
            cite = raw.strip()
            if cite in self._citation_set and cite not in seen:
                seen.add(cite)
                found.append(cite)
        return found

    def get_last_citations(self) -> list[str]:
        """Return citations recovered from the most recent search."""
        return list(self._last_citations)

    def get_last_results(self) -> list[dict]:
        """Return result docs (citation + text + score) from the last search.

        This is the interface ``run_agent`` consumes to pool candidate citations.
        """
        return list(self._last_results)
