import pandas as pd
from tqdm import tqdm

from omnilex.evaluation.scorer import Scorer, validate_submission_format
from our_pipeline.constants import (
    CONFIG,
    COURTS_CSV,
    COURTS_INDEX_PATH,
    DATASET_MODE,
    FORCE_REBUILD_INDICES,
    INDEX_PATH,
    IS_VALIDATION_MODE,
    LAWS_CSV,
    LAWS_INDEX_PATH,
    OUTPUT_PATH,
    QUERY_FILE,
)
from our_pipeline.corpus import get_or_build_index
from our_pipeline.llm.define_agent import run_agent
from our_pipeline.search_tools import CourtSearchTool, LawSearchTool

# === CONFIGURATION ===

# Create output directory
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
INDEX_PATH.mkdir(parents=True, exist_ok=True)

print(f"Environment: 'Local'")
print(f"Dataset mode: {DATASET_MODE}")
print(f"Query file: {QUERY_FILE}")
print(f"Validation mode: {IS_VALIDATION_MODE}")
print(f"Force rebuild indices: {FORCE_REBUILD_INDICES}")
print(f"\nCorpus files:")
print(
    f"  Laws CSV: {LAWS_CSV} ({LAWS_CSV.stat().st_size / 1e6:.1f} MB)"
    if LAWS_CSV.exists()
    else f"  Laws CSV: {LAWS_CSV} (NOT FOUND)"
)
print(
    f"  Courts CSV: {COURTS_CSV} ({COURTS_CSV.stat().st_size / 1e9:.2f} GB)"
    if COURTS_CSV.exists()
    else f"  Courts CSV: {COURTS_CSV} (NOT FOUND)"
)
print(f"\nIndex cache: {INDEX_PATH}")

laws_index = get_or_build_index(
    name="laws",
    csv_path=LAWS_CSV,
    index_path=LAWS_INDEX_PATH,
    force_rebuild=FORCE_REBUILD_INDICES,
    # max_rows=10000  # Uncomment to test with smaller corpus
)

courts_index = get_or_build_index(
    name="courts",
    csv_path=COURTS_CSV,
    index_path=COURTS_INDEX_PATH,
    force_rebuild=FORCE_REBUILD_INDICES,
    # max_rows=100000  # Change to use bigger corpus
)

# Create tools
law_tool = LawSearchTool(
    index=laws_index,
    top_k=CONFIG["top_k_laws"],
    max_excerpt_length=300,
)

court_tool = CourtSearchTool(
    index=courts_index,
    top_k=CONFIG["top_k_courts"],
    max_excerpt_length=300,
)

# Tool registry
TOOLS = {
    "search_laws": law_tool,
    "search_courts": court_tool,
}

print("Tools registered:")
for name, tool in TOOLS.items():
    print(f"  - {name}: {tool.description.split(chr(10))[0]}")


# Load queries from the configured query file
query_file = QUERY_FILE
if not query_file.exists():
    raw_query_file = QUERY_FILE.parent / "raw" / QUERY_FILE.name
    if raw_query_file.exists():
        query_file = raw_query_file
    else:
        raise FileNotFoundError(f"Query file not found: {QUERY_FILE}")

test_df = pd.read_csv(query_file)
print(f"Loaded queries from: {query_file}")

# Generate predictions
predictions = []
all_logs = []  # Store logs for all queries

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running agent"):
    query_id = row["query_id"]
    query_text = row["query"]

    # Run agent
    raw_citations, logs = run_agent(query_text, tools=TOOLS, verbose=False)

    # Store logs with query_id
    all_logs.append({
        "query_id": query_id,
        "query": query_text,
        "logs": logs,
    })

    predictions.append({
        "query_id": query_id,
        "predicted_citations": ";".join(raw_citations),
    })

print(f"\nGenerated predictions for {len(predictions)} queries")
print(f"Collected logs for {len(all_logs)} queries")

predictions_df = pd.DataFrame(predictions)

# Save submission
submission_path = OUTPUT_PATH / "submission.csv"
predictions_df.to_csv(submission_path, index=False)

print(f"Submission saved to: {submission_path}")
print(f"Total predictions: {len(predictions_df)}")

# Show sample
print("\nSample submission:")
print(predictions_df.head())

# Validate and score submission
print("\nValidating submission:")
validation_errors = validate_submission_format(submission_path)
if validation_errors:
    print("Validation failed:")
    for error in validation_errors:
        print(f"  - {error}")
else:
    print("Validation passed")

if "gold_citations" in test_df.columns:
    print(f"\nScoring submission against: {query_file}")
    scores = Scorer().score(submission_path, query_file)

    print("\nScores:")
    for metric, value in scores.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
else:
    print("\nNo gold_citations column found, so scoring is skipped for this file.")
