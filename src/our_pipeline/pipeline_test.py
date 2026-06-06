import pandas as pd
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
from our_pipeline.bm25.corpus import get_or_build_index, get_query_file
from our_pipeline.search_tools import CourtSearchTool, LawSearchTool
from our_pipeline.validation import validate_and_score_submission
from our_pipeline.predictions import generate_predictions

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
query_file = get_query_file()

test_df = pd.read_csv(query_file)
print(f"Loaded queries from: {query_file}")

predictions_df = generate_predictions(test_df, TOOLS)

# Save submission
submission_path = OUTPUT_PATH / "submission.csv"
predictions_df.to_csv(submission_path, index=False)

print(f"Submission saved to: {submission_path}")
print(f"Total predictions: {len(predictions_df)}")

# Show sample
print("\nSample submission:")
print(predictions_df.head())

# Validate and score submission
validate_and_score_submission(query_file)
