import pandas as pd
from datetime import datetime
from our_pipeline.constants import (
    CONFIG,
    COURTS_CSV,
    COURTS_DENSE_META_PATH,
    COURTS_FAISS_PATH,
    COURTS_DENSE_INDEX_DIR,
    COURTS_INDEX_PATH,
    DATASET_MODE,
    FORCE_REBUILD_INDICES,
    FORCE_REBUILD_DENSE,
    INDEX_PATH,
    IS_VALIDATION_MODE,
    LAWS_CSV,
    LAWS_DENSE_INDEX_DIR,
    LAWS_DENSE_META_PATH,
    LAWS_FAISS_PATH,
    LAWS_INDEX_PATH,
    OUTPUT_PATH,
    QUERY_FILE,
)
from our_pipeline.bm25.corpus import get_or_build_index, get_query_file, get_or_build_dense_index
from our_pipeline.dense.index import DenseIndex
from our_pipeline.rerank import make_citation_text_lookup
from our_pipeline.search_tools import CourtSearchTool, DenseSearchTool, LawSearchTool, HybridSearchTool
from our_pipeline.validation import validate_and_score_submission
from our_pipeline.predictions import generate_predictions, save_run_logs

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

# Build dense indices and create hybrid tools when enabled
if CONFIG.get("enable_hybrid_search", False):
    import os
    from dotenv import load_dotenv
    load_dotenv()

    _embed_key = os.getenv("API_KEY")
    _embed_base = CONFIG.get("embedding_api_base", "https://chat-ai.academiccloud.de/v1")
    _embed_model = CONFIG.get("embedding_model", "e5-mistral-7b-instruct")
    _embed_batch = CONFIG.get("embedding_batch_size", 32)

    laws_dense = get_or_build_dense_index(
        name="laws",
        csv_path=LAWS_CSV,
        index_dir=LAWS_DENSE_INDEX_DIR,
        force_rebuild=FORCE_REBUILD_INDICES or FORCE_REBUILD_DENSE,
        max_rows=CONFIG.get("max_docs_dense_laws"),
        model=_embed_model,
        api_key=_embed_key,
        api_base=_embed_base,
        batch_size=_embed_batch,
    )

    courts_dense = get_or_build_dense_index(
        name="courts",
        csv_path=COURTS_CSV,
        index_dir=COURTS_DENSE_INDEX_DIR,
        force_rebuild=FORCE_REBUILD_INDICES or FORCE_REBUILD_DENSE,
        max_rows=CONFIG.get("max_docs_dense_courts", 200_000),
        model=_embed_model,
        api_key=_embed_key,
        api_base=_embed_base,
        batch_size=_embed_batch,
    )

    law_tool = HybridSearchTool(
        bm25_index=laws_index,
        dense_index=laws_dense,
        corpus_type="laws",
        top_k=CONFIG["top_k_laws"],
        max_excerpt_length=300,
    )
    court_tool = HybridSearchTool(
        bm25_index=courts_index,
        dense_index=courts_dense,
        corpus_type="courts",
        top_k=CONFIG["top_k_courts"],
        max_excerpt_length=300,
    )
    print("Hybrid retrieval enabled (BM25 + Dense + RRF + Cohere Rerank)")
else:
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
    print("BM25-only retrieval (set enable_hybrid_search=True for hybrid mode)")

# Tool registry
TOOLS = {
    "search_laws": law_tool,
    "search_courts": court_tool,
}

# Dense (semantic) search tools — registered only when the offline-built FAISS
# artifacts exist
_DENSE_SPECS = [
    (
        "dense_search_laws",
        LAWS_FAISS_PATH,
        LAWS_DENSE_META_PATH,
        CONFIG["top_k_dense_laws"],
        """Semantic search over Swiss federal laws (SR/Systematische Rechtssammlung).
Input: A natural-language legal question or full sentence (any language)
Output: List of relevant law citations with text excerpts

Finds provisions by meaning, even without exact keyword overlap.
""",
    ),
    (
        "dense_search_courts",
        COURTS_FAISS_PATH,
        COURTS_DENSE_META_PATH,
        CONFIG["top_k_dense_courts"],
        """Semantic search over Swiss Federal Court decisions.
Input: A natural-language legal question or full sentence (any language)
Output: List of relevant court decision citations with excerpts

Finds considerations by meaning across languages, without translation.
""",
    ),
]
for _name, _faiss_path, _meta_path, _top_k, _description in _DENSE_SPECS:
    if _faiss_path.exists() and _meta_path.exists():
        TOOLS[_name] = DenseSearchTool(
            index=DenseIndex.load(_faiss_path, _meta_path),
            name=_name,
            description=_description,
            top_k=_top_k,
            max_excerpt_length=300,
        )
    else:
        print(
            f"Warning: dense index for '{_name}' not found ({_faiss_path}). "
            f"Run our_pipeline.dense.build_embeddings; continuing without it."
        )

print("Tools registered:")
for name, tool in TOOLS.items():
    print(f"  - {name}: {tool.description.split(chr(10))[0]}")


# Load queries from the configured query file
query_file = get_query_file()

test_df = pd.read_csv(query_file)
print(f"Loaded queries from: {query_file}")

# # Build court sibling-consideration index (decision -> all its considerations)
# sibling_index = None
# if CONFIG.get("enable_sibling_expansion", True):
#     from expand import build_court_sibling_index

#     sibling_index = build_court_sibling_index(courts_index.documents)
#     print(
#         f"Sibling-consideration index: {len(sibling_index):,} court decisions "
#         f"from {len(courts_index.documents):,} considerations"
#     )

# # Evaluation agent (LQ-RAG precision filter over the recall-maximised pool)
# eval_agent = None
# if CONFIG.get("enable_eval_agent", False):
#     from llm.evaluate_agent import EvaluationAgent

#     # citation -> text lookup spanning both corpora (covers sibling-expanded hits)
#     cite2text: dict[str, str] = {}
#     for doc in laws_index.documents:
#         c = doc.get("citation")
#         if c and c not in cite2text:
#             cite2text[c] = doc.get("text", "")
#     for doc in courts_index.documents:
#         c = doc.get("citation")
#         if c and c not in cite2text:
#             cite2text[c] = doc.get("text", "")

#     eval_agent = EvaluationAgent(cite2text=cite2text)
#     print(f"Evaluation agent enabled (model={CONFIG.get('eval_model')}, "
#           f"cite2text={len(cite2text):,} entries)")

# predictions_df, all_logs = generate_predictions(
#     test_df, TOOLS, sibling_index=sibling_index, eval_agent=eval_agent
# )

# # Persist per-query retrieval logs for debugging recall (decomposition + searches)
# save_run_logs(all_logs, OUTPUT_PATH / "run_logs.jsonl")
# Citation -> full document text, used by the reranker to score (query, document) pairs.
# Laws first, then courts (citation formats don't collide, but order is deterministic).
text_lookup = make_citation_text_lookup(laws_index, courts_index)

predictions_df = generate_predictions(test_df, TOOLS, text_lookup=text_lookup)

# Save submission
submission_path = OUTPUT_PATH / f"submission{str(datetime.now().strftime('%d-%m_%H-%M'))}.csv"
predictions_df.to_csv(submission_path, index=False)

print(f"Submission saved to: {submission_path}")
print(f"Total predictions: {len(predictions_df)}")

# Show sample
print("\nSample submission:")
print(predictions_df.head())

# Validate and score submission
validate_and_score_submission(query_file, submission_path)
