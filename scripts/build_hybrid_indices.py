"""Build BM25 + dense FAISS indices required by the hybrid pipeline.

Run via the SLURM job scripts/run_hybrid_index.sh, or directly:
    python scripts/build_hybrid_indices.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in [str(REPO_ROOT / "src" / "our_pipeline"), str(REPO_ROOT / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    from constants import (
        LAWS_CSV, COURTS_CSV,
        LAWS_INDEX_PATH, COURTS_INDEX_PATH,
        LAWS_DENSE_INDEX_DIR, COURTS_DENSE_INDEX_DIR,
        CONFIG,
    )
    from corpus import get_or_build_index, get_or_build_dense_index

    import torch

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"CUDA available : {torch.cuda.is_available()}  |  GPU count: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  {props.total_memory // 1024**3} GB")

    # ── BM25 indices ──────────────────────────────────────────────────────────────
    print("\n=== Building BM25 indices ===")
    laws_bm25   = get_or_build_index("laws",   LAWS_CSV,   LAWS_INDEX_PATH)
    courts_bm25 = get_or_build_index("courts", COURTS_CSV, COURTS_INDEX_PATH)
    print(f"Laws BM25  : {len(laws_bm25.documents):,} documents")
    print(f"Courts BM25: {len(courts_bm25.documents):,} documents")

    # ── Dense indices ─────────────────────────────────────────────────────────────
    LOCAL_EMBED_MODEL = str(REPO_ROOT / CONFIG["embedding_model"])
    BATCH_SIZE        = CONFIG.get("embedding_batch_size", 64)
    MAX_LAWS_ROWS     = CONFIG.get("max_docs_dense_laws")
    MAX_COURTS_ROWS   = CONFIG.get("max_docs_dense_courts", 200_000)

    print(f"\n=== Building dense indices ===")
    print(f"Embedding model : {LOCAL_EMBED_MODEL}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Max laws rows   : {MAX_LAWS_ROWS}")
    print(f"Max courts rows : {MAX_COURTS_ROWS:,}")

    laws_dense = get_or_build_dense_index(
        name="laws",
        csv_path=LAWS_CSV,
        index_dir=LAWS_DENSE_INDEX_DIR,
        max_rows=MAX_LAWS_ROWS,
        model=LOCAL_EMBED_MODEL,
        batch_size=BATCH_SIZE,
    )
    print(f"Laws dense  : {len(laws_dense.documents):,} documents")

    courts_dense = get_or_build_dense_index(
        name="courts",
        csv_path=COURTS_CSV,
        index_dir=COURTS_DENSE_INDEX_DIR,
        max_rows=MAX_COURTS_ROWS,
        model=LOCAL_EMBED_MODEL,
        batch_size=BATCH_SIZE,
    )
    print(f"Courts dense: {len(courts_dense.documents):,} documents")

    print("\n=== All indices ready ===")
