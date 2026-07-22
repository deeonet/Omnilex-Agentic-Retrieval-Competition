"""Download all local models used by the LQ-RAG pipeline into ``models/``.

Run once on a node with internet (Slurm: add ``-C inet``).  Subsequent jobs
reference the local snapshot paths and need no network access.

Usage:
    python scripts/download_models.py                 # all default models
    python scripts/download_models.py --only embedding # just the embedder
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

# role -> (HF repo id, local dir name under models/)
MODELS = {
    "embedding": ("intfloat/multilingual-e5-large", "multilingual-e5-large"),
    "reranker": ("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3"),
    "generator": ("Qwen/Qwen2.5-14B-Instruct", "qwen2.5-14b-instruct"),
    # Smaller fallback if 14B does not fit alongside embedder/reranker.
    "generator_small": ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        choices=list(MODELS),
        default=["embedding", "reranker", "generator"],
        help="Subset of roles to download (default: embedding reranker generator).",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for role in args.only:
        repo_id, local_name = MODELS[role]
        target = MODELS_DIR / local_name
        print(f"\n=== {role}: {repo_id} -> {target} ===")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
        print(f"Done: {target}")

    print("\nAll requested models downloaded.")


if __name__ == "__main__":
    main()
