from pathlib import Path
# Choose which dataset to run on: "val" or "test"
DATASET_MODE = "val"  # Change to "test" for final submission

# Set to True to rebuild indices from CSV (required on first run)
# Set to False to load cached indices (faster for subsequent runs)
FORCE_REBUILD_INDICES = False


# Local development paths
# Resolved from this file's location (src/our_pipeline/constants.py) so that
# batch jobs launched from any working directory find the same artifacts.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "raw"
MODEL_PATH = REPO_ROOT / "models"
OUTPUT_PATH = REPO_ROOT / "output"
INDEX_PATH = REPO_ROOT / "data" / "processed"

# CSV corpus files for index building
LAWS_CSV = DATA_PATH / "laws_de.csv"
COURTS_CSV = DATA_PATH / "court_considerations.csv"

# Index cache paths
LAWS_INDEX_PATH = INDEX_PATH / "laws_index.pkl"
COURTS_INDEX_PATH = INDEX_PATH / "courts_index.pkl"

# Dense (embedding) index artifacts, built by our_pipeline.dense.build_embeddings
DENSE_DIR = INDEX_PATH / "dense"
DENSE_SHARDS_DIR = DENSE_DIR / "shards"
LAWS_FAISS_PATH = DENSE_DIR / "laws.faiss"
LAWS_DENSE_META_PATH = DENSE_DIR / "laws_meta.parquet"
COURTS_FAISS_PATH = DENSE_DIR / "courts.faiss"
COURTS_DENSE_META_PATH = DENSE_DIR / "courts_meta.parquet"

# Derived paths based on DATASET_MODE
QUERY_FILE = DATA_PATH / f"{DATASET_MODE}.csv"
IS_VALIDATION_MODE = DATASET_MODE == "val"

CONFIG = {
    # Model settings
    "model_file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "n_ctx": 8192,         # Context window size
    "n_threads": 4,
    "n_gpu_layers": -1,    # GPU layers (-1 = offload all layers to GPU)
    
    # Agent settings
    # Non-reasoning instruct model for agent calls: the default reasoning model
    # (qwen3.5-27b) burns the whole token budget on hidden <think> on knowledge-heavy
    # prompts and the API returns empty content -> 0 citations for that query.
    "agent_model": "qwen3-30b-a3b-instruct-2507",
    "max_iterations": 3,   # Max agent iterations per query
    "max_tokens": 512,
    "max_tokens_retry": 1024,  # Larger budget when a call returns empty content
    "temperature": 0.1,
    "max_observation_chars": 1200,  # Reduced from 2000 to prevent context overflow
    "max_conversation_chars": 28000,  # Safety net: truncate if conversation exceeds this
    
    # Retrieval settings
    # How predictions are produced per query:
    #   "recall_union" - directly call ALL tools and return their RRF-fused union
    #                    (recall-first; deterministic; guarantees every tool incl. dense contributes)
    #   "agent"        - ReAct agent returns only its curated "Final Answer" set (precision-first)
    "retrieval_mode": "recall_union",
    # Cap on citations returned per query after RRF fusion (recall_union mode).
    # Set to None to measure the uncapped recall ceiling of the retrievers.
    "max_predictions": 160,
    "top_k_laws": 30,       # Results per law search
    "top_k_courts": 30,     # Results per court search
    "enable_multilingual_search": True,  # Court tool only: translate to EN/DE/FR/IT (corpus is multilingual)
    "translation_model": "qwen3-30b-a3b-instruct-2507",  # non-reasoning instruct model

    # BM25 index/tokenization settings (changing any of these requires an index rebuild)
    "bm25_k1": 1.5,                 # BM25Okapi term-frequency saturation
    "bm25_b": 0.75,                 # BM25Okapi length normalization
    "bm25_remove_stopwords": True,  # Drop German + legal stopwords during tokenization
    "bm25_use_stemming": True,      # Snowball German stemming (no-op if snowballstemmer is unavailable)

    # Law tool query strategy (German-only corpus -> German keyword extraction + explicit citations,
    # fused with Reciprocal Rank Fusion). When False, search the raw query directly.
    "law_query_expansion": True,
    "rrf_k": 60,                    # Reciprocal Rank Fusion constant

    # Dense retrieval (Qwen3-Embedding-4B + FAISS; build offline with
    # our_pipeline.dense.build_embeddings, see scripts/embed_corpus.sbatch)
    "embed_model": "Qwen/Qwen3-Embedding-4B",
    "embed_dim": 2560,              # Native output dim (MRL allows truncation on rebuild)
    "top_k_dense_laws": 50,         # Results per dense law search
    "top_k_dense_courts": 50,       # Results per dense court search
    "dense_over_retrieve": 4,       # Over-retrieval factor before dedup by citation
    "embed_batch_size": 64,         # Encode batch size during the offline build
    "embed_max_seq_laws": 2048,     # Token cap per law article during the build
    "embed_max_seq_courts": 1024,   # Token cap per court consideration during the build

    # Reranking (Qwen3-Reranker-4B CrossEncoder; filters the fused union for precision).
    # Each retrieved citation's full document text is scored against the query, the raw
    # logit is squashed to a [0,1] confidence (sigmoid), and only citations clearing the
    # threshold are kept. Trades recall for precision -> tune on val to maximize F1.
    "enable_reranking": True,
    "reranker_model": "Qwen/Qwen3-Reranker-4B",
    "reranker_threshold": 0.5,      # Keep citations with sigmoid confidence >= this (TUNE on val)
    "reranker_min_keep": 10,         # Recall floor: if nothing clears the threshold, keep this many top-scored
    "reranker_batch_size": 32,      # Pairs per CrossEncoder.predict batch
    "reranker_max_length": 1024,    # Token cap per (query, document) pair (court texts are long)

    # Paths
    "test_file": "test.csv",
}
