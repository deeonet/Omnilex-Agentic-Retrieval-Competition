# Omnilex Agentic Retrieval — Swiss Legal Citation Retrieval

Kaggle competition: [llm-agentic-legal-information-retrieval](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval)

Given a natural-language legal query the system must return the exact set of relevant citations — Swiss federal law articles and Federal Court decisions — that appear verbatim in the corpus. The competition metric is **Macro F1** over citation sets.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data Setup](#data-setup)
4. [Environment Variables](#environment-variables)
5. [Pipeline Modes](#pipeline-modes)
   - [Mode 1 — BM25 Only](#mode-1--bm25-only)
   - [Mode 2 — Hybrid (BM25 + Dense + RRF + Cohere Rerank)](#mode-2--hybrid-bm25--dense--rrf--cohere-rerank)
   - [Mode 3 — GraphRAG](#mode-3--graphrag)
6. [Evaluation](#evaluation)
7. [Key Configuration Reference](#key-configuration-reference)
8. [Project Structure](#project-structure)
9. [Notebooks](#notebooks)
10. [Kaggle Submission](#kaggle-submission)

---

## Prerequisites

- Python >= 3.10
- A Kaggle account (competition joined, API token generated)
- Access to the Academic Cloud LLM endpoint (`chat-ai.academiccloud.de`)
- *For Hybrid mode:* A [Mistral AI](https://console.mistral.ai/) API key and a [Cohere](https://cohere.com/) API key
- *For GraphRAG mode:* A Mistral AI API key

---

## Installation

```bash
# Clone the team fork
git clone git@github.com:deeonet/Omnilex-Agentic-Retrieval-Competition.git
cd Omnilex-Agentic-Retrieval-Competition

# Create a personal dev branch
git checkout -b your-name.dev

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install runtime + dev dependencies
pip install -r requirements-dev.txt

# Install the shared omnilex library in editable mode
pip install -e .
```

---

## Data Setup

```bash
# Set the Kaggle token (generated at https://www.kaggle.com/settings → API)
export KAGGLE_API_TOKEN=KGAT_your_token_here

# Download competition data (opens browser for login on first run)
python download_kaggle_data.py

# Copy files to data/ (the script prints the cache path)
cp -r ~/.cache/kagglehub/competitions/llm-agentic-legal-information-retrieval/versions/*/* data/
```

Expected files in `data/`:

| File | Size | Description |
|------|------|-------------|
| `train.csv` | — | Training queries (non-English) with `gold_citations` |
| `val.csv` | — | 10 English validation queries with `gold_citations` |
| `test.csv` | — | 40 English test queries (no labels) |
| `laws_de.csv` | — | SR corpus: law articles in German |
| `court_considerations.csv` | ~2.4 GB | BGE corpus: court decisions in DE/FR/IT |
| `sample_submission.csv` | — | Submission format reference |

---

## Environment Variables

All runtime secrets are stored in a `.env` file at the **repo root**. Create it before running any pipeline:

```bash
cp .env.example .env   # if provided, otherwise create from scratch
```

### Root `.env` — used by the custom pipeline (`src/our_pipeline/`)

```dotenv
# ── LLM (Academic Cloud, required for all modes) ────────────────────────────
API_KEY=your_academic_cloud_api_key
API_BASE_URL=https://chat-ai.academiccloud.de/v1   # optional, this is the default
API_MODEL=qwen3.5-27b                               # optional, this is the default

# ── Mistral AI embeddings (required for Hybrid and GraphRAG modes) ───────────
# Free tier available at https://console.mistral.ai/ — no credit card needed
MISTRAL_API_KEY=your_mistral_api_key

# ── Cohere Rerank (required for Hybrid mode) ────────────────────────────────
# Sign up at https://cohere.com/ — free trial available
COHERE_API_KEY=your_cohere_api_key
```

### `graphrag/.env` — used exclusively by the GraphRAG indexing & query pipeline

```dotenv
# Academic Cloud LLM (for entity extraction, summarisation, community reports)
GRAPHRAG_API_KEY=your_academic_cloud_api_key
GRAPHRAG_API_BASE=https://chat-ai.academiccloud.de/v1
GRAPHRAG_CHAT_MODEL=qwen3.5-27b

# Mistral AI embeddings (for vector store)
MISTRAL_API_KEY=your_mistral_api_key
```

> **Note:** `MISTRAL_API_KEY` must appear in **both** `.env` files if you intend to run the Hybrid pipeline and the GraphRAG pipeline from the same environment.

---

## Pipeline Modes

All modes run from inside the `src/our_pipeline/` directory:

```bash
cd src/our_pipeline
```

The active mode is set in [`src/our_pipeline/constants.py`](src/our_pipeline/constants.py):

```python
DATASET_MODE = "val"   # "val" for local scoring, "test" for final submission
```

---

### Mode 1 — BM25 Only

Lexical keyword matching with `rank-bm25` (k1=1.2, b=0.75). Queries are translated to EN/DE/FR/IT and fused via CombMAX.

**No additional API keys required beyond `API_KEY`.**

**Step 1 — Disable hybrid mode in `constants.py`:**

```python
CONFIG = {
    ...
    "enable_hybrid_search": False,
    ...
}
```

**Step 2 — Build BM25 indices (first run only):**

Indices are cached automatically on first run. To force a rebuild set `FORCE_REBUILD_INDICES = True` in `constants.py`.

**Step 3 — Run the pipeline:**

```bash
python pipeline_test.py
```

Outputs `output/submission.csv` and prints Macro F1 when `DATASET_MODE="val"`.

---

### Mode 2 — Hybrid (BM25 + Dense + RRF + Cohere Rerank)

Three-stage pipeline:

1. **BM25** (multilingual CombMAX) → top-50 candidates
2. **Dense retrieval** (Mistral `mistral-embed` + FAISS) → top-50 candidates
3. **RRF fusion** (k=60) → merged top-50 list
4. **Cohere Rerank** (`rerank-multilingual-v3.0`) → final top-10

**Required keys in `.env`:** `API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`

**Step 1 — Enable hybrid mode in `constants.py`:**

```python
CONFIG = {
    ...
    "enable_hybrid_search": True,
    ...
}
```

**Step 2 — Install additional dependencies:**

```bash
pip install faiss-cpu cohere
# On GPU clusters use faiss-gpu for faster index builds:
# pip install faiss-gpu
```

**Step 3 — Build all indices (first run only):**

BM25 indices are built as before. Dense indices are built automatically on first run, calling the Mistral embeddings API in batches.

Estimated build times (Mistral free tier, 60 req/min):

| Index | Docs | Approx. time |
|-------|------|--------------|
| `laws_dense_index/` | all (~5k) | ~5 min |
| `courts_dense_index/` | 200k (default) | ~60 min |

Dense indices are cached to `data/processed/laws_dense_index/` and `data/processed/courts_dense_index/`. Subsequent runs load from cache instantly.

To index more court rows, change `max_docs_dense_courts` in `constants.py`:

```python
"max_docs_dense_courts": 500_000,   # increase from default 200_000
```

To force a rebuild of dense indices, set `FORCE_REBUILD_INDICES = True` in `constants.py`.

**Step 4 — Run the pipeline:**

```bash
python pipeline_test.py
```

**Graceful fallback:** If `COHERE_API_KEY` is missing, the tool uses the RRF-fused results directly. If the dense index is missing, it falls back to BM25 only.

---

### Mode 3 — GraphRAG

Knowledge-graph-based retrieval using Microsoft GraphRAG. Entities and relationships are extracted from the corpus, clustered into communities, and searched via local or global search.

**Required keys:** `GRAPHRAG_API_KEY`, `MISTRAL_API_KEY` (in `graphrag/.env`)

**Step 1 — Prepare text files for GraphRAG:**

```bash
# Laws only (faster, ~10 MB of text)
python scripts/prepare_graphrag_data.py --laws-only

# Laws + a subset of court decisions
python scripts/prepare_graphrag_data.py --max-court-excerpts 200000
```

This writes plain-text files to `graphrag/input/`.

**Step 2 — Add your keys to `graphrag/.env`:**

```bash
cat > graphrag/.env << 'EOF'
GRAPHRAG_API_KEY=your_academic_cloud_api_key
GRAPHRAG_API_BASE=https://chat-ai.academiccloud.de/v1
GRAPHRAG_CHAT_MODEL=qwen3.5-27b
MISTRAL_API_KEY=your_mistral_api_key
EOF
```

**Step 3 — Build the knowledge graph index (one-time, slow):**

```bash
graphrag index --root ./graphrag
```

This performs entity extraction, relationship mapping, community detection, and embedding. The resulting artefacts are written to `graphrag/output/`.

**Step 4 — Use GraphRAG in your pipeline:**

The shared library exposes ready-to-use tools:

```python
from omnilex.retrieval.graphrag_tools import GraphRAGLocalSearchTool, GraphRAGGlobalSearchTool

local_tool  = GraphRAGLocalSearchTool(root="./graphrag")
global_tool = GraphRAGGlobalSearchTool(root="./graphrag")
```

See `notebooks/03_graphrag_retrieval.ipynb` for a full worked example.

---

## Evaluation

```bash
# Score the latest submission against the val set
python scripts/evaluate_submission.py output/submission.csv

# Score against train set
python scripts/evaluate_submission.py output/submission.csv --split train

# Per-query breakdown
python scripts/evaluate_submission.py output/submission.csv -v
```

`pipeline_test.py` also auto-scores when `DATASET_MODE="val"` and the query file contains a `gold_citations` column.

Run the unit tests with:

```bash
pytest
pytest tests/test_citations/test_normalizer.py -v   # single file
```

---

## Key Configuration Reference

All knobs live in [`src/our_pipeline/constants.py`](src/our_pipeline/constants.py).

| Key | Default | Description |
|-----|---------|-------------|
| `DATASET_MODE` | `"val"` | `"val"` for local scoring, `"test"` for final submission |
| `FORCE_REBUILD_INDICES` | `False` | Rebuild all BM25 and dense indices from CSV |
| `enable_hybrid_search` | `True` | Use Hybrid mode; `False` for BM25-only |
| `enable_multilingual_search` | `True` | Translate queries to EN/DE/FR/IT before BM25 |
| `top_k_laws` | `10` | Final law results returned to the agent per search |
| `top_k_courts` | `10` | Final court results returned to the agent per search |
| `max_iterations` | `3` | ReAct loop iterations per query |
| `dense_candidate_k` | `50` | Candidates per retriever before RRF |
| `rrf_k` | `60` | RRF smoothing constant |
| `cohere_rerank_candidates` | `50` | Candidates sent to Cohere reranker |
| `cohere_rerank_model` | `"rerank-multilingual-v3.0"` | Cohere rerank model |
| `embedding_model` | `"mistral-embed"` | Embedding model for dense index |
| `embedding_batch_size` | `32` | Documents per embedding API call |
| `max_docs_dense_courts` | `200_000` | Court rows to embed (full corpus is 2.4 M) |
| `max_docs_dense_laws` | `None` | Law rows to embed (`None` = all) |
| `max_observation_chars` | `1200` | Truncation limit for tool output sent to LLM |

---

## Project Structure

```
Omnilex-Agentic-Retrieval-Competition/
│
├── .env                            # Root secrets (API_KEY, MISTRAL_API_KEY, COHERE_API_KEY)
├── requirements.txt                # Runtime dependencies
├── requirements-dev.txt            # Test/lint dependencies
│
├── src/
│   ├── omnilex/                    # Shared competition library (pip install -e .)
│   │   ├── citations/              # CitationNormalizer — parses/validates citation strings
│   │   ├── evaluation/             # macro_f1, micro_f1, MAP, NDCG; Scorer class
│   │   ├── retrieval/              # BM25Index, GraphRAG search tools
│   │   └── llm/                    # LLMLoader, prompt templates
│   │
│   └── our_pipeline/               # Team's custom submission pipeline
│       ├── constants.py            # ← All config knobs live here
│       ├── corpus.py               # BM25Index, DenseIndex, index builders
│       ├── rerank.py               # reciprocal_rank_fusion, cohere_rerank
│       ├── search_tools.py         # LawSearchTool, CourtSearchTool, HybridSearchTool
│       ├── predictions.py          # generate_predictions — runs agent over queries
│       ├── pipeline_test.py        # ← Main entrypoint: builds indices, runs pipeline
│       ├── validation.py           # validate_and_score_submission helper
│       └── llm/
│           ├── load_llm.py         # OpenAICompatibleLLM adapter (reads .env)
│           ├── translate.py        # translate_query — EN/DE/FR/IT via LLM
│           ├── define_agent.py     # run_agent — ReAct loop (Mistral Instruct format)
│           └── prompts.py          # AGENT_SYSTEM_PROMPT
│
├── graphrag/                       # GraphRAG configuration and prompts
│   ├── .env                        # GraphRAG secrets (GRAPHRAG_API_KEY, MISTRAL_API_KEY)
│   ├── settings.yaml               # LLM, embedding, chunking, entity type settings
│   └── prompts/                    # Custom extraction / summarisation prompts
│
├── notebooks/
│   ├── 01_direct_generation_baseline.ipynb   # LLM generates citations directly
│   ├── 02_agentic_retrieval_baseline.ipynb   # ReAct + BM25 (omnilex library)
│   ├── 03_agentic_retrieval_custom.ipynb     # Custom pipeline walkthrough
│   └── 03_graphrag_retrieval.ipynb           # GraphRAG integration demo
│
├── data/
│   ├── train.csv / val.csv / test.csv
│   ├── laws_de.csv                 # SR corpus (German)
│   ├── court_considerations.csv   # BGE corpus (DE/FR/IT, ~2.4 GB)
│   └── processed/
│       ├── laws_index.pkl          # Cached BM25 index (auto-built)
│       ├── courts_index.pkl
│       ├── laws_dense_index/       # Cached FAISS index (auto-built in Hybrid mode)
│       └── courts_dense_index/
│
├── scripts/
│   ├── evaluate_submission.py
│   └── prepare_graphrag_data.py
│
└── tests/
    └── test_citations/
```

---

## Notebooks

| Notebook | Mode | Description |
|----------|------|-------------|
| `01_direct_generation_baseline.ipynb` | LLM direct | LLM generates citations from memory — fast but hallucinates |
| `02_agentic_retrieval_baseline.ipynb` | BM25 agent | ReAct agent with BM25 tools using the `omnilex` library |
| `03_agentic_retrieval_custom.ipynb` | Custom pipeline | Walkthrough of `src/our_pipeline/` end-to-end |
| `03_graphrag_retrieval.ipynb` | GraphRAG | GraphRAG local/global search demo |

---

## Kaggle Submission

1. Set `DATASET_MODE = "test"` in `constants.py`
2. Run `python src/our_pipeline/pipeline_test.py`
3. Submit `output/submission.csv` to Kaggle

The submission format is:

```csv
query_id,predicted_citations
test_001,"Art. 11 Abs. 2 OR;BGE 139 I 2 E. 3.1"
test_002,"5A_800/2019 E 5."
test_003,""
```

Citations must be **exact verbatim strings** from the corpus. Anything outside the closed citation vocabulary scores as a false positive.

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

## Contact

For public competition questions use the Kaggle Discussion tab or open an issue on this repository.  
For private questions: ari.jordan@omnilex.ai
