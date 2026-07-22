"""LightRAG retrieval pipeline (laws-first).

Same decompose-loop pipeline as pipeline_test.py, but the *laws* search tool is
the graph-augmented LightRAGSearchTool instead of BM25. The court tool stays
BM25 (+ sibling expansion) so end-to-end val scores stay comparable.

Prerequisites:
  - LightRAG laws index built: sbatch scripts/run_lightrag_index.sh
    (creates lightrag/laws/ — JSON KV + NanoVectorDB + NetworkX, no parquet)
  - Qwen (vLLM) completion server + e5 embedding server reachable at the URLs
    in lightrag_config (LightRAG runs query-time keyword extraction via the LLM).
  - The agent LLM (constants/.env) must also be reachable.

Run (servers must be up — see scripts/run_lightrag_index.sh for how to start them):
  cd <repo_root>
  PYTHONPATH=src/our_pipeline:src python src/our_pipeline/pipeline_lightrag.py
"""
import os
import sys
from pathlib import Path

# ── import order: our_pipeline modules first (no package install needed) ──────
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd

from constants import (
    CONFIG,
    COURTS_CSV,
    COURTS_INDEX_PATH,
    DATASET_MODE,
    FORCE_REBUILD_INDICES,
    INDEX_PATH,
    IS_VALIDATION_MODE,
    LAWS_CSV,
    OUTPUT_PATH,
    QUERY_FILE,
)
from corpus import get_or_build_index, get_query_file
from search_tools import CourtSearchTool
from predictions import generate_predictions, save_run_logs
from validation import validate_and_score_submission

from omnilex.retrieval.lightrag_tools import LightRAGSearchTool

LIGHTRAG_LAWS_DIR = Path(
    os.environ.get("LIGHTRAG_LAWS_WORKDIR", str(Path(__file__).parent.parent.parent / "lightrag" / "laws"))
)

# ── sanity-check: LightRAG laws index must exist ──────────────────────────────
if not LIGHTRAG_LAWS_DIR.exists() or not any(LIGHTRAG_LAWS_DIR.iterdir()):
    print(f"ERROR: LightRAG laws index not found at {LIGHTRAG_LAWS_DIR}")
    print("  Run: sbatch scripts/run_lightrag_index.sh")
    sys.exit(1)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
INDEX_PATH.mkdir(parents=True, exist_ok=True)

print("=== LightRAG Pipeline (laws-first) ===")
print(f"Dataset mode    : {DATASET_MODE}")
print(f"Query file      : {QUERY_FILE}")
print(f"LightRAG laws   : {LIGHTRAG_LAWS_DIR}")

# ── Tools: LightRAG for laws, BM25 for courts ─────────────────────────────────
print("\nLoading LightRAG laws knowledge graph…")
law_tool = LightRAGSearchTool(
    workdir=LIGHTRAG_LAWS_DIR,
    top_k=CONFIG["top_k_laws"],
)

courts_index = get_or_build_index(
    name="courts",
    csv_path=COURTS_CSV,
    index_path=COURTS_INDEX_PATH,
    force_rebuild=FORCE_REBUILD_INDICES,
)
court_tool = CourtSearchTool(
    index=courts_index,
    top_k=CONFIG["top_k_courts"],
    max_excerpt_length=300,
)

TOOLS = {
    "search_laws": law_tool,
    "search_courts": court_tool,
}

print("\nTools registered:")
for name, tool in TOOLS.items():
    print(f"  - {name}: {tool.description.strip().splitlines()[0]}")

# ── Court sibling-consideration index (decision -> all considerations) ────────
sibling_index = None
if CONFIG.get("enable_sibling_expansion", True):
    from expand import build_court_sibling_index

    sibling_index = build_court_sibling_index(courts_index.documents)
    print(
        f"Sibling-consideration index: {len(sibling_index):,} court decisions "
        f"from {len(courts_index.documents):,} considerations"
    )

# ── Evaluation agent (LQ-RAG precision filter over the recall-maximised pool) ─
eval_agent = None
if CONFIG.get("enable_eval_agent", False):
    from llm.evaluate_agent import EvaluationAgent

    # citation -> text lookup spanning both corpora (covers sibling-expanded hits).
    # Laws come straight from the CSV — this pipeline has no laws BM25 index.
    laws_df = pd.read_csv(LAWS_CSV, usecols=["citation", "text"]).dropna()
    cite2text: dict[str, str] = {}
    for c, t in zip(laws_df["citation"].astype(str), laws_df["text"].astype(str)):
        if c not in cite2text:
            cite2text[c] = t
    for doc in courts_index.documents:
        c = doc.get("citation")
        if c and c not in cite2text:
            cite2text[c] = doc.get("text", "")

    eval_agent = EvaluationAgent(cite2text=cite2text)
    print(f"Evaluation agent enabled (model={CONFIG.get('eval_model')}, "
          f"cite2text={len(cite2text):,} entries)")

# ── Run agent over queries ────────────────────────────────────────────────────
query_file = get_query_file()
test_df = pd.read_csv(query_file)
print(f"\nLoaded {len(test_df)} queries from {query_file}")

predictions_df, all_logs = generate_predictions(
    test_df, TOOLS, sibling_index=sibling_index, eval_agent=eval_agent
)

# ── Save and score ────────────────────────────────────────────────────────────
# Write submission.csv too — validate_and_score_submission() scores that path.
predictions_df.to_csv(OUTPUT_PATH / "submission.csv", index=False)
submission_path = OUTPUT_PATH / "submission_lightrag.csv"
predictions_df.to_csv(submission_path, index=False)
print(f"\nSubmission saved to: {submission_path} (and submission.csv for scoring)")
print(predictions_df.head())

save_run_logs(all_logs, OUTPUT_PATH / "run_logs_lightrag.jsonl")

if IS_VALIDATION_MODE:
    validate_and_score_submission(query_file)
