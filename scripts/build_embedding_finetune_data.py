"""Build (query -> gold-citation-text) pairs for embedding fine-tuning (LQ-RAG FT Layer).

Positives are the corpus text of each gold citation for each train query — these are
the real cross-lingual pairs (multilingual query -> DE/FR/IT statute or ruling text)
that the fine-tune teaches the embedder to align.

Streams the large court CSV keeping only rows whose citation is actually cited by
some train query, so it stays memory-light.

Outputs (under data/processed/embed_ft/):
    train_pairs.jsonl   {"query": ..., "positive": ...}   (MNRL anchor/positive)
    eval_corpus.json    {doc_id: text}      held-out IR eval corpus
    eval_queries.json   {qid: query}
    eval_qrels.json     {qid: [doc_id, ...]}
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = DATA / "processed" / "embed_ft"

csv.field_size_limit(10_000_000)


def split_gold(cell: str) -> list[str]:
    if not isinstance(cell, str):
        return []
    return [c.strip() for c in cell.split(";") if c.strip()]


def load_citation_text(needed: set[str]) -> dict[str, str]:
    """Map citation -> text for every needed citation, from both corpora."""
    lookup: dict[str, str] = {}

    laws = pd.read_csv(DATA / "laws_de.csv")
    for cit, txt in zip(laws["citation"], laws["text"]):
        if cit in needed and cit not in lookup:
            lookup[str(cit)] = str(txt)

    # Stream courts (2.4 GB) — keep only needed citations.
    with open(DATA / "court_considerations.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cit = row.get("citation", "")
            if cit in needed and cit not in lookup:
                lookup[cit] = row.get("text", "")
    return lookup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-frac", type=float, default=0.1, help="Fraction of queries held out for IR eval.")
    ap.add_argument("--distractors", type=int, default=2000, help="Random corpus distractors added to eval corpus.")
    ap.add_argument("--max-chars", type=int, default=1200, help="Truncate positive/corpus texts.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    tr = pd.read_csv(DATA / "train.csv")
    tr = tr.dropna(subset=["query", "gold_citations"]).reset_index(drop=True)
    print(f"Loaded {len(tr)} train queries.")

    needed: set[str] = set()
    for cell in tr["gold_citations"]:
        needed.update(split_gold(cell))
    print(f"Resolving text for {len(needed)} unique gold citations…")

    lookup = load_citation_text(needed)
    resolved = len(lookup)
    print(f"Resolved {resolved}/{len(needed)} citations to corpus text "
          f"({resolved / max(len(needed),1):.1%} coverage).")

    def clip(t: str) -> str:
        return t[: args.max_chars]

    # Held-out split by query index.
    idx = list(range(len(tr)))
    rng.shuffle(idx)
    n_eval = int(len(idx) * args.eval_frac)
    eval_idx = set(idx[:n_eval])

    train_pairs: list[dict] = []
    eval_queries: dict[str, str] = {}
    eval_qrels: dict[str, list[str]] = {}
    eval_corpus: dict[str, str] = {}
    heldout_rows: list[dict] = []  # for end-to-end pipeline scoring (never trained on)

    for i, row in tr.iterrows():
        q = str(row["query"]).strip()
        golds = [g for g in split_gold(row["gold_citations"]) if g in lookup]
        if not golds:
            continue
        if i in eval_idx:
            qid = f"q{i}"
            eval_queries[qid] = clip(q)
            eval_qrels[qid] = golds
            for g in golds:
                eval_corpus[g] = clip(lookup[g])
            # Preserve original query_id + full (unclipped) gold string for scoring.
            heldout_rows.append({
                "query_id": row.get("query_id", qid),
                "query": q,
                "gold_citations": str(row["gold_citations"]),
            })
        else:
            for g in golds:
                train_pairs.append({"query": q, "positive": clip(lookup[g])})

    # Add random distractors to the eval corpus so Recall/MRR are meaningful.
    pool = [c for c in lookup if c not in eval_corpus]
    rng.shuffle(pool)
    for c in pool[: args.distractors]:
        eval_corpus[c] = clip(lookup[c])

    with open(OUT / "train_pairs.jsonl", "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    pd.DataFrame(heldout_rows).to_csv(OUT / "heldout_queries.csv", index=False)
    for name, obj in [
        ("eval_corpus.json", eval_corpus),
        ("eval_queries.json", eval_queries),
        ("eval_qrels.json", eval_qrels),
    ]:
        with open(OUT / name, "w") as f:
            json.dump(obj, f, ensure_ascii=False)

    print(f"\nWrote {len(train_pairs)} training pairs, "
          f"{len(eval_queries)} eval queries, {len(eval_corpus)} eval corpus docs, "
          f"{len(heldout_rows)} held-out scoring rows to {OUT}")


if __name__ == "__main__":
    main()
