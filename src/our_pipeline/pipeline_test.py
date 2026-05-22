import os
import sys
from pathlib import Path
from our_pipeline.constants import (
    OUTPUT_PATH, INDEX_PATH, DATASET_MODE, QUERY_FILE, IS_VALIDATION_MODE, FORCE_REBUILD_INDICES, LAWS_CSV, COURTS_CSV, CONFIG, LAWS_INDEX_PATH, COURTS_INDEX_PATH
    )
from our_pipeline.corpus import get_or_build_index
from our_pipeline.search_tools import LawSearchTool, CourtSearchTool

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
print(f"  Laws CSV: {LAWS_CSV} ({LAWS_CSV.stat().st_size / 1e6:.1f} MB)" if LAWS_CSV.exists() else f"  Laws CSV: {LAWS_CSV} (NOT FOUND)")
print(f"  Courts CSV: {COURTS_CSV} ({COURTS_CSV.stat().st_size / 1e9:.2f} GB)" if COURTS_CSV.exists() else f"  Courts CSV: {COURTS_CSV} (NOT FOUND)")
print(f"\nIndex cache: {INDEX_PATH}")


laws_index = get_or_build_index(
    name="laws",
    csv_path=LAWS_CSV,
    index_path=LAWS_INDEX_PATH,
    force_rebuild=FORCE_REBUILD_INDICES,
    # max_rows=10000  # Uncomment to test with smaller corpus
)
print(f"\nLaws index: {len(laws_index.documents):,} documents")

# Test search
test_results = laws_index.search("Vertrag", top_k=3)
print(f"\nTest search 'Vertrag': {len(test_results)} results")
if test_results:
    print(f"  Top result: {test_results[0].get('citation', 'N/A')}")


courts_index = get_or_build_index(
    name="courts",
    csv_path=COURTS_CSV,
    index_path=COURTS_INDEX_PATH,
    force_rebuild=FORCE_REBUILD_INDICES,
    # max_rows=100000  # Change to use bigger corpus
)
print(f"\nCourts index: {len(courts_index.documents):,} documents")

# Test search
test_results = courts_index.search("Meinungsfreiheit", top_k=3)
print(f"\nTest search 'Meinungsfreiheit': {len(test_results)} results")
if test_results:
    print(f"  Top result: {test_results[0].get('citation', 'N/A')}")


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

# Test tools
print("Testing law search:")
print(law_tool("Vertrag Abschluss"))

print("\nTesting court search:")
print(court_tool("Meinungsfreiheit"))