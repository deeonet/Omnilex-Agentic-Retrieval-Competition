from pathlib import Path
# Choose which dataset to run on: "val" or "test"
DATASET_MODE = "val"  # Change to "test" for final submission

# Set to True to rebuild indices from CSV (required on first run)
# Set to False to load cached indices (faster for subsequent runs)
FORCE_REBUILD_INDICES = False


# Local development paths
REPO_ROOT = Path(".").resolve()
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
    "max_iterations": 3,   # Max agent iterations per query
    "max_tokens": 512,
    "temperature": 0.1,
    "max_observation_chars": 1200,  # Reduced from 2000 to prevent context overflow
    "max_conversation_chars": 28000,  # Safety net: truncate if conversation exceeds this
    
    # Retrieval settings
    "top_k_laws": 40,       # Results per law search
    "top_k_courts": 40,     # Results per court search
    "enable_multilingual_search": True,  # Translate queries to EN+DE+FR before BM25 (CombMAX fusion)
    
    # Paths
    "test_file": "test.csv",
}
