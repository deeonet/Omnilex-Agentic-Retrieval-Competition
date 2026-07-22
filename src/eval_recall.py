"""Recall@k harness for the Law search tool over the validation set.

Measures how many of the *retrievable* law citations (the gold citations that actually
exist in ``laws_de.csv``) the ``LawSearchTool`` surfaces in its top-k. Court citations in
the gold set (BGE / docket numbers) are ignored — the law tool cannot serve them.

Usage:
    python src/eval_recall.py                # all val queries, current CONFIG
    python src/eval_recall.py --limit 25     # first 25 queries
    python src/eval_recall.py --raw          # disable keyword expansion (BM25 on raw query)
    python src/eval_recall.py --dense        # dense (embedding) search instead of BM25
    python src/eval_recall.py --show-missed   # also print the gold law citations that were missed
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

from our_pipeline.constants import (
    CONFIG,
    DATA_PATH,
    FORCE_REBUILD_INDICES,
    LAWS_CSV,
    LAWS_DENSE_META_PATH,
    LAWS_FAISS_PATH,
    LAWS_INDEX_PATH,
)
from our_pipeline.bm25.corpus import get_or_build_index
from our_pipeline.search_tools import DenseSearchTool, LawSearchTool


def clean(citation: str) -> str:
    """Normalize whitespace and strip trailing separators for exact-match comparison.

    Gold, corpus, and predicted citations share the same surface format
    (e.g. "Art. 221 Abs. 1 StPO"); this only collapses spacing and removes stray
    trailing ``;``/``:`` (as seen for entries like "Art. 63 Abs. 1 SpoFöV:").
    """
    return re.sub(r"\s+", " ", citation).strip().strip(";: ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall@k for the law search tool.")
    parser.add_argument("--limit", type=int, default=None, help="First N queries only.")
    parser.add_argument("--dataset", default="val", help="Dataset CSV stem in data/raw.")
    parser.add_argument("--raw", action="store_true", help="Disable expansion (raw query).")
    parser.add_argument("--dense", action="store_true", help="Use the dense (embedding) law index instead of BM25.")
    parser.add_argument("--show-missed", action="store_true", help="Print missed gold citations.")
    args = parser.parse_args()

    if args.raw:
        CONFIG["law_query_expansion"] = False

    df = pd.read_csv(DATA_PATH / f"{args.dataset}.csv")
    if args.limit:
        df = df.head(args.limit)

    if args.dense:
        from our_pipeline.dense.index import DenseIndex

        dense_index = DenseIndex.load(LAWS_FAISS_PATH, LAWS_DENSE_META_PATH)
        # Citation universe from the dense metadata (same CSV rows as the BM25
        # build) so we don't load both indices just for the retrievability filter.
        corpus_citations = {clean(c) for c in dense_index.meta["citation"]}
        law_tool = DenseSearchTool(
            index=dense_index,
            name="dense_search_laws",
            description="Semantic search over Swiss federal laws.",
            top_k=CONFIG["top_k_dense_laws"],
            max_excerpt_length=300,
        )
        top_k = CONFIG["top_k_dense_laws"]
        expansion = "dense embeddings"
    else:
        index = get_or_build_index(
            name="laws",
            csv_path=LAWS_CSV,
            index_path=LAWS_INDEX_PATH,
            force_rebuild=FORCE_REBUILD_INDICES,
        )
        corpus_citations = {clean(doc["citation"]) for doc in index.documents}
        law_tool = LawSearchTool(index=index, top_k=CONFIG["top_k_laws"], max_excerpt_length=300)
        top_k = CONFIG["top_k_laws"]
        expansion = "raw query" if args.raw else "keyword expansion + RRF"

    print(f"\nrecall@{top_k} on {len(df)} '{args.dataset}' queries ({expansion})\n")

    recalls: list[float] = []
    for _, row in df.iterrows():
        gold = [clean(g) for g in str(row["gold_citations"]).split(";") if g.strip()]
        gold_law = [g for g in gold if g in corpus_citations]
        if not gold_law:
            continue  # nothing retrievable from the law corpus -> skip

        law_tool(row["query"])
        predicted = {clean(c) for c in law_tool.get_last_citations()}

        hits = [g for g in gold_law if g in predicted]
        recall = len(hits) / len(gold_law)
        recalls.append(recall)

        line = f"{row['query_id']}: recall {recall:.2f}  ({len(hits)}/{len(gold_law)} law cites)"
        if args.show_missed:
            missed = [g for g in gold_law if g not in predicted]
            if missed:
                line += f"\n    missed: {missed}"
        print(line)

    if recalls:
        print(f"\nMean recall@{top_k}: {sum(recalls) / len(recalls):.3f} "
              f"over {len(recalls)} queries with retrievable law gold.")
    else:
        print("\nNo queries had retrievable law gold citations.")


if __name__ == "__main__":
    main()
