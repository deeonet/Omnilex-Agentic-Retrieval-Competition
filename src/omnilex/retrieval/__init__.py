"""Retrieval tools and indexing for Swiss legal documents."""

try:
    from .bm25_index import BM25Index, build_index, load_jsonl_corpus, search
except ImportError:
    BM25Index = build_index = load_jsonl_corpus = search = None  # rank_bm25 not installed

from .graphrag_tools import GraphRAGGlobalSearchTool, GraphRAGLocalSearchTool, GraphRAGSearchEngine

try:
    from .tools import CourtSearchTool, LawSearchTool
except ImportError:
    CourtSearchTool = LawSearchTool = None

__all__ = [
    "BM25Index",
    "build_index",
    "load_jsonl_corpus",
    "search",
    "LawSearchTool",
    "CourtSearchTool",
    "GraphRAGLocalSearchTool",
    "GraphRAGGlobalSearchTool",
    "GraphRAGSearchEngine",
]
