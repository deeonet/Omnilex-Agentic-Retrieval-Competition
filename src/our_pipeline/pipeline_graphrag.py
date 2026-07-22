"""GraphRAG-only retrieval pipeline.

Uses GraphRAGLocalSearchTool exclusively — no BM25 fallback.

Prerequisites:
  - GraphRAG index must be built: sbatch scripts/run_graphrag_index.sh
  - graphrag/output/*.parquet files must exist (entities, communities,
    community_reports, text_units, relationships)
  - LLM must be reachable (local GGUF or API via .env)

Run:
  cd <repo_root>
  conda run -n mypython python src/our_pipeline/pipeline_graphrag.py
"""
import sys
from pathlib import Path

# ── import order: our_pipeline modules first (no package install needed) ──────
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd

from constants import (
    DATASET_MODE,
    IS_VALIDATION_MODE,
    OUTPUT_PATH,
    QUERY_FILE,
)
from corpus import get_query_file
from predictions import generate_predictions, save_run_logs
from validation import validate_and_score_submission

from omnilex.retrieval.graphrag_tools import GraphRAGLocalSearchTool

GRAPHRAG_ROOT = Path(__file__).parent.parent.parent / "graphrag"
OUTPUT_DIR = GRAPHRAG_ROOT / "output"

# ── sanity-check: index must exist ────────────────────────────────────────────
REQUIRED_PARQUETS = [
    "entities", "communities", "community_reports", "text_units", "relationships"
]
missing = [f for f in REQUIRED_PARQUETS if not (OUTPUT_DIR / f"{f}.parquet").exists()]
if missing:
    print(f"ERROR: GraphRAG index incomplete. Missing: {missing}")
    print(f"  Run: sbatch scripts/run_graphrag_index.sh")
    sys.exit(1)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("=== GraphRAG-Only Pipeline ===")
print(f"Dataset mode : {DATASET_MODE}")
print(f"Query file   : {QUERY_FILE}")
print(f"GraphRAG root: {GRAPHRAG_ROOT}")

# ── GraphRAG local-search tool ────────────────────────────────────────────────
print("\nLoading GraphRAG knowledge graph (this may take a minute for large graphs)…")
graphrag_tool = GraphRAGLocalSearchTool(
    graphrag_root=GRAPHRAG_ROOT,
    community_level=2,
    response_type="Single Paragraph",
)

TOOLS = {
    "graphrag_search": graphrag_tool,
}

print("\nTools registered:")
for name, tool in TOOLS.items():
    print(f"  - {name}: {tool.description.strip().splitlines()[0]}")

# ── Run agent over queries ─────────────────────────────────────────────────────
query_file = get_query_file()
test_df = pd.read_csv(query_file)
print(f"\nLoaded {len(test_df)} queries from {query_file}")

predictions_df, all_logs = generate_predictions(test_df, TOOLS)

# ── Save and score ─────────────────────────────────────────────────────────────
submission_path = OUTPUT_PATH / "submission_graph.csv"
predictions_df.to_csv(submission_path, index=False)
print(f"\nSubmission saved to: {submission_path}")
print(predictions_df.head())

save_run_logs(all_logs, OUTPUT_PATH / "run_logs_graph.jsonl")

if IS_VALIDATION_MODE:
    validate_and_score_submission(query_file)
