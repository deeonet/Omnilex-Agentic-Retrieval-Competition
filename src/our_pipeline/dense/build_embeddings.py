"""Offline build of the dense FAISS indices for the legal corpora.

Two phases:

1. **Embed** (default; needs a GPU for realistic corpus sizes): encode corpus
   rows with Qwen3-Embedding-4B into fp16 ``.npy`` shards. Resumable — shards
   that already exist are skipped, so rerunning after a job failure only does
   the missing work. Shards keep the full native dim so the index can later be
   rebuilt at a lower MRL dim without re-embedding (truncate + re-normalize).
   SLURM array jobs split the shards automatically via SLURM_ARRAY_TASK_ID /
   SLURM_ARRAY_TASK_COUNT (or pass --worker-id/--num-workers explicitly).

2. **Assemble** (``--assemble``; CPU-only): verify all shards exist, build the
   FAISS index (laws: exact FlatIP; courts: exact fp16 scalar quantizer at half
   the memory) and write the positionally aligned metadata parquet.

Usage:
    python -m our_pipeline.dense.build_embeddings --corpus laws
    python -m our_pipeline.dense.build_embeddings --corpus laws --assemble
    sbatch scripts/embed_corpus.sbatch laws

Smoke test (CPU, ~2 min):
    python -m our_pipeline.dense.build_embeddings --corpus laws \\
        --max-rows 500 --batch-size 4 --shard-size 200 --out-dir data/processed/dense_smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from our_pipeline.bm25.corpus import load_csv_corpus
from our_pipeline.constants import CONFIG, COURTS_CSV, DENSE_DIR, LAWS_CSV

CORPUS_CSV = {"laws": LAWS_CSV, "courts": COURTS_CSV}


def compose_embed_text(doc: dict) -> str:
    """Text that gets embedded: title + body for laws, body for courts.

    The citation string is deliberately excluded — citation tokens carry no
    semantic signal, and explicit-citation queries are already served exactly
    by ``BM25Index.get_by_citation`` in the law tool.
    """
    title = doc.get("title", "")
    text = doc.get("text", "")
    return f"{title}\n{text}" if title else text


def shard_file(out_dir: Path, corpus: str, shard_idx: int) -> Path:
    return out_dir / "shards" / corpus / f"shard_{shard_idx:05d}.npy"


def manifest_file(out_dir: Path, corpus: str) -> Path:
    return out_dir / f"{corpus}_manifest.json"


def shard_bounds(shard_idx: int, shard_size: int, total: int) -> tuple[int, int]:
    return shard_idx * shard_size, min((shard_idx + 1) * shard_size, total)


def run_embed(args: argparse.Namespace, docs: list[dict]) -> None:
    total = len(docs)
    num_shards = math.ceil(total / args.shard_size)
    my_shards = [
        i
        for i in range(num_shards)
        if i % args.num_workers == args.worker_id and not shard_file(args.out_dir, args.corpus, i).exists()
    ]
    skipped = sum(
        1 for i in range(num_shards)
        if i % args.num_workers == args.worker_id and shard_file(args.out_dir, args.corpus, i).exists()
    )
    print(
        f"[embed] corpus={args.corpus} rows={total:,} shards={num_shards} "
        f"worker={args.worker_id}/{args.num_workers} todo={len(my_shards)} resumed/skipped={skipped}"
    )

    manifest = {
        "model": CONFIG["embed_model"],
        "dim": CONFIG["embed_dim"],
        "normalized": True,
        "shard_size": args.shard_size,
        "total_rows": total,
        "csv_name": CORPUS_CSV[args.corpus].name,
        "max_rows": args.max_rows,
        "max_seq_length": args.max_seq_length,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = manifest_file(args.out_dir, args.corpus)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, manifest_path)

    if not my_shards:
        print("[embed] nothing to do for this worker.")
        return

    from our_pipeline.dense.embedder import load_embedding_model

    model = load_embedding_model(max_seq_length=args.max_seq_length)
    print(f"[embed] model loaded on {model.device}, max_seq_length={model.max_seq_length}")

    shard_file(args.out_dir, args.corpus, 0).parent.mkdir(parents=True, exist_ok=True)
    for n, shard_idx in enumerate(my_shards, 1):
        start, end = shard_bounds(shard_idx, args.shard_size, total)
        print(f"[embed] shard {shard_idx} ({n}/{len(my_shards)}): rows {start:,}-{end:,}")
        texts = [compose_embed_text(doc) for doc in docs[start:end]]
        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float16)

        path = shard_file(args.out_dir, args.corpus, shard_idx)
        tmp = path.with_suffix(".npy.tmp")
        with open(tmp, "wb") as f:  # atomic write: a killed job never leaves a truncated shard
            np.save(f, embeddings)
        os.replace(tmp, path)
    print(f"[embed] worker {args.worker_id} done.")


def run_assemble(args: argparse.Namespace) -> None:
    import faiss
    import pandas as pd

    manifest_path = manifest_file(args.out_dir, args.corpus)
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path} — run the embed phase first.")
    manifest = json.loads(manifest_path.read_text())

    total = manifest["total_rows"]
    shard_size = manifest["shard_size"]
    dim = manifest["dim"]
    num_shards = math.ceil(total / shard_size)

    missing = [i for i in range(num_shards) if not shard_file(args.out_dir, args.corpus, i).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{num_shards} shards missing for '{args.corpus}': {missing[:20]}"
            f"{' ...' if len(missing) > 20 else ''}. Rerun the embed phase to fill them in."
        )

    if args.corpus == "laws":
        index = faiss.IndexFlatIP(dim)
    else:
        # Exact search at half the memory of FlatIP (fp16-quantized storage).
        index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)

    print(f"[assemble] corpus={args.corpus} rows={total:,} shards={num_shards} dim={dim}")
    for shard_idx in range(num_shards):
        start, end = shard_bounds(shard_idx, shard_size, total)
        arr = np.load(shard_file(args.out_dir, args.corpus, shard_idx))
        if arr.shape != (end - start, dim):
            raise ValueError(
                f"Shard {shard_idx} has shape {arr.shape}, expected {(end - start, dim)} — "
                f"stale shard from a different build? Delete it and re-embed."
            )
        arr = arr.astype(np.float32)
        if shard_idx == 0 and not index.is_trained:
            index.train(arr)
        index.add(arr)
        print(f"[assemble] added shard {shard_idx + 1}/{num_shards} (ntotal={index.ntotal:,})")

    faiss_path = args.out_dir / f"{args.corpus}.faiss"
    faiss.write_index(index, str(faiss_path))
    print(f"[assemble] wrote {faiss_path} ({faiss_path.stat().st_size / 1e9:.2f} GB)")

    # Metadata rows align positionally with FAISS ids because both come from the
    # same load_csv_corpus call (CSV row order) with the same max_rows.
    docs = load_csv_corpus(CORPUS_CSV[args.corpus], max_rows=manifest["max_rows"])
    if len(docs) != total:
        raise ValueError(
            f"CSV now yields {len(docs):,} rows but the shards were built from {total:,} — "
            f"the corpus file changed since embedding. Re-embed before assembling."
        )
    meta = {
        "citation": [doc["citation"] for doc in docs],
        "excerpt": [doc["text"][: args.excerpt_chars] for doc in docs],
    }
    if any("title" in doc for doc in docs):
        meta["title"] = [doc.get("title", "") for doc in docs]
    meta_path = args.out_dir / f"{args.corpus}_meta.parquet"
    pd.DataFrame(meta).to_parquet(meta_path, compression="zstd", index=False)
    print(f"[assemble] wrote {meta_path} ({meta_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[assemble] done: {index.ntotal:,} vectors.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense FAISS indices for the legal corpora.")
    parser.add_argument("--corpus", required=True, choices=["laws", "courts"])
    parser.add_argument("--assemble", action="store_true", help="Build FAISS index + metadata from completed shards.")
    parser.add_argument("--batch-size", type=int, default=CONFIG["embed_batch_size"])
    parser.add_argument("--max-seq-length", type=int, default=None,
                        help="Token cap per document (default: CONFIG embed_max_seq_<corpus>).")
    parser.add_argument("--max-rows", type=int, default=None, help="Row limit for smoke tests.")
    parser.add_argument("--shard-size", type=int, default=50_000, help="Rows per .npy shard.")
    parser.add_argument("--worker-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    parser.add_argument("--out-dir", type=Path, default=DENSE_DIR,
                        help="Artifact directory (override for smoke tests to protect the real index).")
    parser.add_argument("--excerpt-chars", type=int, default=500, help="Chars of text kept in the metadata.")
    args = parser.parse_args()

    if args.max_seq_length is None:
        args.max_seq_length = CONFIG[f"embed_max_seq_{args.corpus}"]

    if args.assemble:
        run_assemble(args)
    else:
        docs = load_csv_corpus(CORPUS_CSV[args.corpus], max_rows=args.max_rows)
        run_embed(args, docs)


if __name__ == "__main__":
    main()
