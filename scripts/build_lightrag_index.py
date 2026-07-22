#!/usr/bin/env python
"""Build a LightRAG graph index over the Swiss legal corpus (laws first).

Each corpus row is inserted so that its citation is recoverable from retrieved
context two ways (belt-and-suspenders):
  1. the citation is prefixed into the inserted text as ``[citation]`` so it
     survives into retrieved chunk text (regex-recoverable downstream), and
  2. it is passed as the LightRAG ``file_paths`` source for the document.

Requires the Qwen (vLLM) completion server and the e5 embedding server to be
running — see scripts/run_lightrag_index.sh. LightRAG writes its index as plain
local files under the working dir (no parquet / pyarrow).

Usage:
    python scripts/build_lightrag_index.py --laws
    python scripts/build_lightrag_index.py --laws --limit 50        # smoke test
    python scripts/build_lightrag_index.py --laws --workdir lightrag/laws_smoke --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from omnilex.retrieval.lightrag_config import make_lightrag  # noqa: E402

LAWS_CSV = REPO_ROOT / "data" / "laws_de.csv"
DEFAULT_LAWS_WORKDIR = REPO_ROOT / "lightrag" / "laws"


def _load_laws(limit: int | None, codes: list[str] | None = None) -> list[tuple[str, str]]:
    """Return [(citation, text)] for the laws corpus, deduped by citation.

    ``codes`` restricts to citations whose trailing SR-number token matches one
    of the given codes (e.g. ``["OR", "ZGB", "220"]``) — useful for building a
    small, targeted index instead of the full ~176k-row corpus.
    """
    df = pd.read_csv(LAWS_CSV)
    df = df.dropna(subset=["citation", "text"])
    df = df.drop_duplicates(subset=["citation"], keep="first")
    if codes:
        code_set = set(codes)
        df = df[df["citation"].str.split().str[-1].isin(code_set)]
    if limit is not None:
        df = df.head(limit)
    return list(zip(df["citation"].astype(str), df["text"].astype(str)))


async def _build(
    rows: list[tuple[str, str]], workdir: Path, batch_size: int, max_async: int | None
) -> None:
    overrides = {"llm_model_max_async": max_async} if max_async else {}
    rag = await make_lightrag(workdir, **overrides)
    total = len(rows)
    print(f"Inserting {total:,} documents into {workdir} (batch={batch_size})", flush=True)
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        texts = [f"[{cite}]\n{text}" for cite, text in batch]
        ids = [cite for cite, _ in batch]
        file_paths = [cite for cite, _ in batch]
        await rag.ainsert(texts, ids=ids, file_paths=file_paths)
        print(f"  inserted {min(start + batch_size, total):,}/{total:,}", flush=True)
    # Flush any pending storage state to disk.
    if hasattr(rag, "finalize_storages"):
        await rag.finalize_storages()
    print("Done.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laws", action="store_true", help="Build the laws index")
    parser.add_argument("--limit", type=int, default=None, help="Only insert the first N rows (smoke test)")
    parser.add_argument("--workdir", type=str, default=None, help="LightRAG working dir")
    parser.add_argument("--batch-size", type=int, default=200, help="Docs per insert batch")
    parser.add_argument(
        "--codes",
        type=str,
        default=None,
        help="Comma-separated SR codes to restrict to (e.g. 'OR,ZGB,StGB'), matched "
        "against each citation's trailing token. Default: all codes.",
    )
    parser.add_argument(
        "--max-async",
        type=int,
        default=None,
        help="Override LightRAG's llm_model_max_async (default in LightRAG is 4).",
    )
    args = parser.parse_args()

    if not args.laws:
        parser.error("Nothing to do — pass --laws (court support comes later).")

    workdir = Path(args.workdir) if args.workdir else DEFAULT_LAWS_WORKDIR
    codes = [c.strip() for c in args.codes.split(",")] if args.codes else None
    rows = _load_laws(args.limit, codes=codes)
    asyncio.run(_build(rows, workdir, args.batch_size, args.max_async))


if __name__ == "__main__":
    main()
