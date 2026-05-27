"""GraphRAG-powered search tools for legal citation retrieval.

Wraps Microsoft GraphRAG's local and global search APIs.
Requires a completed graphrag index (run `graphrag index --root ./graphrag`).

Two search modes:
- Local search: best for specific citation/provision lookups (entity-centric)
- Global search: best for broad thematic questions (community-based summaries)
"""

import asyncio
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
GRAPHRAG_ROOT = PROJECT_ROOT / "graphrag"
OUTPUT_DIR = GRAPHRAG_ROOT / "output"


def _load_output_tables(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Load GraphRAG output parquet files into DataFrames."""
    required = {
        "entities": output_dir / "entities.parquet",
        "communities": output_dir / "communities.parquet",
        "community_reports": output_dir / "community_reports.parquet",
        "text_units": output_dir / "text_units.parquet",
        "relationships": output_dir / "relationships.parquet",
    }
    tables: dict[str, pd.DataFrame] = {}
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"GraphRAG output file not found: {path}\n"
                "Run `graphrag index --root ./graphrag` to build the index first."
            )
        tables[name] = pd.read_parquet(path)
    return tables


def _load_config(graphrag_root: Path):
    """Load GraphRAG config from settings.yaml."""
    from graphrag.config.load_config import load_config

    return load_config(graphrag_root)


class GraphRAGLocalSearchTool:
    """Entity-centric local search over the GraphRAG knowledge graph.

    Best for queries about specific legal provisions, articles, or citations.
    Uses vector similarity + graph neighborhood traversal.
    """

    name: str = "graphrag_local_search"
    description: str = """Search the legal knowledge graph using entity-centric local search.
                        Input: A specific legal question or citation to look up
                        Output: Detailed answer drawing on related law articles, court decisions, and legal concepts

                        Use this when you need to find specific provisions or understand how particular
                        legal articles relate to each other in the knowledge graph.
                        Example queries: "Art. 1 OR contract formation", "BGE 119 II 449 negligence"
                        """

    def __init__(
        self,
        graphrag_root: Path | str = GRAPHRAG_ROOT,
        community_level: int = 2,
        response_type: str = "Single Paragraph",
    ):
        self.graphrag_root = Path(graphrag_root)
        self.community_level = community_level
        self.response_type = response_type
        self._tables: dict[str, pd.DataFrame] | None = None
        self._config = None

    def _ensure_loaded(self) -> None:
        if self._tables is None:
            self._tables = _load_output_tables(self.graphrag_root / "output")
        if self._config is None:
            self._config = _load_config(self.graphrag_root)

    def __call__(self, query: str) -> str:
        return self.run(query)

    def run(self, query: str) -> str:
        if not query or not query.strip():
            return "Error: Empty query."
        self._ensure_loaded()
        return asyncio.run(self._run_async(query))

    async def _run_async(self, query: str) -> str:
        from graphrag.api import local_search

        result, _ = await local_search(
            config=self._config,
            entities=self._tables["entities"],
            communities=self._tables["communities"],
            community_reports=self._tables["community_reports"],
            text_units=self._tables["text_units"],
            relationships=self._tables["relationships"],
            covariates=None,
            community_level=self.community_level,
            response_type=self.response_type,
            query=query,
        )
        return str(result) if not isinstance(result, str) else result

    def search_with_metadata(self, query: str):
        """Return (answer, context_data) tuple for inspection."""
        self._ensure_loaded()

        async def _inner():
            from graphrag.api import local_search

            return await local_search(
                config=self._config,
                entities=self._tables["entities"],
                communities=self._tables["communities"],
                community_reports=self._tables["community_reports"],
                text_units=self._tables["text_units"],
                relationships=self._tables["relationships"],
                covariates=None,
                community_level=self.community_level,
                response_type=self.response_type,
                query=query,
            )

        return asyncio.run(_inner())


class GraphRAGGlobalSearchTool:
    """Community-based global search over the GraphRAG knowledge graph.

    Best for broad thematic questions that require synthesising across
    many legal provisions (e.g. "What are the grounds for contract invalidity?").
    Uses hierarchical community summaries.
    """

    name: str = "graphrag_global_search"
    description: str = """Search the legal knowledge graph using community-based global search.
Input: A broad legal question or topic
Output: Synthesised answer from community summaries across the knowledge graph

Use this for high-level questions that span multiple laws or legal areas.
Example queries: "grounds for contract invalidity in Swiss law",
                 "requirements for environmental impact assessment"
"""

    def __init__(
        self,
        graphrag_root: Path | str = GRAPHRAG_ROOT,
        community_level: int | None = None,
        dynamic_community_selection: bool = True,
        response_type: str = "Multiple Paragraphs",
    ):
        self.graphrag_root = Path(graphrag_root)
        self.community_level = community_level
        self.dynamic_community_selection = dynamic_community_selection
        self.response_type = response_type
        self._tables: dict[str, pd.DataFrame] | None = None
        self._config = None

    def _ensure_loaded(self) -> None:
        if self._tables is None:
            self._tables = _load_output_tables(self.graphrag_root / "output")
        if self._config is None:
            self._config = _load_config(self.graphrag_root)

    def __call__(self, query: str) -> str:
        return self.run(query)

    def run(self, query: str) -> str:
        if not query or not query.strip():
            return "Error: Empty query."
        self._ensure_loaded()
        return asyncio.run(self._run_async(query))

    async def _run_async(self, query: str) -> str:
        from graphrag.api import global_search

        result, _ = await global_search(
            config=self._config,
            entities=self._tables["entities"],
            communities=self._tables["communities"],
            community_reports=self._tables["community_reports"],
            community_level=self.community_level,
            dynamic_community_selection=self.dynamic_community_selection,
            response_type=self.response_type,
            query=query,
        )
        return str(result) if not isinstance(result, str) else result


class GraphRAGSearchEngine:
    """Combined search engine that integrates GraphRAG with BM25 retrieval.

    Provides a unified interface over all search tools for the agentic pipeline.
    Automatically picks local vs. global search based on query characteristics.
    """

    def __init__(
        self,
        graphrag_root: Path | str = GRAPHRAG_ROOT,
        bm25_law_tool=None,
        bm25_court_tool=None,
    ):
        self.graphrag_root = Path(graphrag_root)
        self.local_tool = GraphRAGLocalSearchTool(graphrag_root)
        self.global_tool = GraphRAGGlobalSearchTool(graphrag_root)
        self.bm25_law_tool = bm25_law_tool
        self.bm25_court_tool = bm25_court_tool

    def get_all_tools(self) -> list:
        """Return all tools as a list for use in a ReAct agent."""
        tools = [self.local_tool, self.global_tool]
        if self.bm25_law_tool is not None:
            tools.append(self.bm25_law_tool)
        if self.bm25_court_tool is not None:
            tools.append(self.bm25_court_tool)
        return tools

    def hybrid_search(self, query: str, use_bm25: bool = True) -> str:
        """Run local GraphRAG search and optionally augment with BM25 results."""
        parts = []

        graph_result = self.local_tool.run(query)
        parts.append("=== Knowledge Graph (Local Search) ===")
        parts.append(graph_result)

        if use_bm25:
            if self.bm25_law_tool is not None:
                parts.append("\n=== Federal Laws (BM25) ===")
                parts.append(self.bm25_law_tool.run(query))
            if self.bm25_court_tool is not None:
                parts.append("\n=== Court Decisions (BM25) ===")
                parts.append(self.bm25_court_tool.run(query))

        return "\n".join(parts)
