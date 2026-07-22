import os
from pathlib import Path

def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    return default if val is None else val.strip().lower() in ("1", "true", "yes")

# Choose which dataset to run on: "val" or "test" (env override: DATASET_MODE)
DATASET_MODE = os.getenv("DATASET_MODE", "val")  # Change to "test" for final submission

# Set to True to rebuild the BM25 indices from CSV (slow for the 2.4 GB court
# corpus). Env override: FORCE_REBUILD_INDICES.
FORCE_REBUILD_INDICES = _env_bool("FORCE_REBUILD_INDICES", False)

# Rebuild ONLY the dense (FAISS) indices — needed after swapping the embedding
# model, without paying the cost of rebuilding BM25. Env: FORCE_REBUILD_DENSE.
FORCE_REBUILD_DENSE = _env_bool("FORCE_REBUILD_DENSE", False)


# Local development paths
REPO_ROOT = Path(".").resolve()
DATA_PATH = REPO_ROOT / "data"
MODEL_PATH = REPO_ROOT / "models"
OUTPUT_PATH = REPO_ROOT / "output"
INDEX_PATH = REPO_ROOT / "data" / "processed"

# CSV corpus files for index building
LAWS_CSV = DATA_PATH / "laws_de.csv"
COURTS_CSV = DATA_PATH / "court_considerations.csv"

# BM25 index cache paths
LAWS_INDEX_PATH = INDEX_PATH / "laws_index.pkl"
COURTS_INDEX_PATH = INDEX_PATH / "courts_index.pkl"

# Dense (FAISS) index cache directories
LAWS_DENSE_INDEX_DIR = INDEX_PATH / "laws_dense_index"
COURTS_DENSE_INDEX_DIR = INDEX_PATH / "courts_dense_index"

# Derived paths based on DATASET_MODE.
# QUERY_FILE env override lets the scoring sbatch point at an arbitrary CSV
# (e.g. a held-out train slice with gold_citations for inline Macro-F1).
QUERY_FILE = Path(os.getenv("QUERY_FILE")) if os.getenv("QUERY_FILE") else DATA_PATH / f"{DATASET_MODE}.csv"
IS_VALIDATION_MODE = DATASET_MODE == "val"

CONFIG = {
    # Model settings
    "model_file": "",  # empty = skip local GGUF, use API fallback
    "n_ctx": 8192,         # Context window size
    "n_threads": 4,
    "n_gpu_layers": -1,    # GPU layers (-1 = offload all layers to GPU)
    
    # Agent settings
    "max_iterations": 3,   # Legacy ReAct loop; unused by deterministic decompose-loop
    "max_tokens": 512,
    "temperature": 0.1,
    "max_observation_chars": 1200,  # Reduced from 2000 to prevent context overflow
    "max_conversation_chars": 28000,  # Safety net: truncate if conversation exceeds this
    
    # Query decomposition (deterministic decompose-loop)
    "max_decompose_issues": 12,   # Max focused sub-issues per query
    "decompose_max_tokens": 1024,
    "decompose_model": "qwen3-30b-a3b-instruct-2507",  # non-reasoning instruct model

    # Sibling-consideration expansion (court decisions)
    "enable_sibling_expansion": True,   # Add all considerations of each retrieved decision
    "max_siblings_per_decision": None,  # Cap siblings added per decision (None = all)

    # Retrieval settings
    "top_k_laws": 20,        # Final results returned per law sub-issue search
    "top_k_courts": 20,      # Final results returned per court sub-issue search
    "enable_multilingual_search": True,  # Translate queries to EN+DE+FR before BM25 (CombMAX fusion)
    "translation_model": "qwen3-30b-a3b-instruct-2507",  # non-reasoning instruct model

    # Hybrid retrieval (BM25 + Dense + RRF + Rerank)
    "enable_hybrid_search": True,        # Use HybridSearchTool instead of BM25-only tools
    "dense_candidate_k": 50,             # Candidates retrieved per retriever before RRF
    "rrf_k": 60,                         # RRF smoothing constant (original paper default)
    "cohere_rerank_candidates": 50,      # Candidates sent to the reranker
    # Reranker backend: "local" (BGE cross-encoder, offline) | "cohere" | "none"
    "rerank_backend": os.getenv("RERANK_BACKEND", "local"),
    "local_rerank_model": os.getenv("LOCAL_RERANK_MODEL", "models/bge-reranker-v2-m3"),
    "cohere_rerank_model": "rerank-multilingual-v3.0",  # Cohere model (multilingual: DE/FR/IT)

    # Evaluation agent (LQ-RAG RAG-triad precision filter over candidate citations)
    "enable_eval_agent": _env_bool("ENABLE_EVAL_AGENT", True),
    "eval_model": os.getenv("EVAL_MODEL", "qwen3-30b-a3b-instruct-2507"),
    "eval_batch_size": 10,        # candidates judged per LLM call
    "eval_max_candidates": 80,    # cap candidates judged per query (latency bound)
    "eval_excerpt_chars": 400,    # citation text shown to the judge
    "eval_max_tokens": 256,
    "eval_fallback_top_k": 10,    # kept if the judge rejects everything (protects recall)

    # Dense index settings
    "embedding_model": os.getenv("EMBEDDING_MODEL", "models/multilingual-e5-large"),  # local dir
    "embedding_api_base": "https://chat-ai.academiccloud.de/v1",  # unused when local
    "embedding_batch_size": 64,  # larger batches on GPU
    # Limit court dense index size (full 2.4 M rows is ~10 GB; set None for all rows)
    "max_docs_dense_courts": 200_000,
    "max_docs_dense_laws": None,         # Laws corpus is small — index all rows

    # Paths
    "test_file": "test.csv",
}
