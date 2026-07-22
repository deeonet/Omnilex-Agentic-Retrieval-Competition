#!/usr/bin/env python3
"""
Diagnose retrieval recall from the candidate pool, *before* any selection step.

The deterministic decompose-loop (src/our_pipeline) writes per-query retrieval
logs to output/run_logs.jsonl. This script answers the key question that
separates a retrieval problem from a selection problem:

    Of the gold citations, how many were retrieved into the candidate pool at all?

It reports:
  - recall@candidates  (macro + micro): gold ∩ pool / gold
  - oracle Macro F1    : the F1 ceiling if selection were perfect
                         (i.e. you predicted exactly gold ∩ pool)
  - per-query breakdown and, in verbose mode, the gold citations that were
    missed entirely (never retrieved — a recall problem no selector can fix).

Comparing oracle F1 here against the actual Macro F1 from evaluate_submission.py
tells you where to invest next: a low oracle F1 means push retrieval (P0/P2);
a high oracle F1 but low actual F1 means build/improve the selection stage (P1).

Usage:
    python scripts/recall_at_candidates.py                      # output/run_logs.jsonl vs data/val.csv
    python scripts/recall_at_candidates.py path/to/run_logs.jsonl --split train
    python scripts/recall_at_candidates.py -v                   # show missed gold per query
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Set

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_LOGS = PROJECT_ROOT / "output" / "run_logs.jsonl"

_WS_RE = re.compile(r"\s+")


def _canon(c: str) -> str:
    """Match evaluate_submission.py: strip + collapse internal whitespace."""
    return _WS_RE.sub(" ", c.strip())


def _parse_gold(value: object, sep: str = ";") -> Set[str]:
    if pd.isna(value):
        return set()
    if not isinstance(value, str):
        value = str(value)
    return {c for c in (_canon(p) for p in value.split(sep)) if c}


def _pool_from_logs(entry: dict) -> Set[str]:
    """Extract the candidate pool (set of canonical citations) for one query.

    Prefers the 'summary' log's deduped citation list; falls back to the union
    of all 'search' entries if no summary is present.
    """
    logs = entry.get("logs", [])
    pool: Set[str] = set()
    saw_summary = False
    for log in logs:
        if log.get("type") == "summary":
            saw_summary = True
            for c in log.get("citations", []):
                if c:
                    pool.add(_canon(str(c)))
    if not saw_summary:
        for log in logs:
            if log.get("type") == "search":
                for c in log.get("citations", []):
                    if c:
                        pool.add(_canon(str(c)))
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute recall@candidates and the oracle Macro F1 ceiling from run_logs.jsonl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "logs",
        type=Path,
        nargs="?",
        default=DEFAULT_LOGS,
        help=f"Path to run_logs.jsonl (default: {DEFAULT_LOGS})",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Gold split to evaluate against (default: val)",
    )
    parser.add_argument(
        "--solution", "-s",
        type=Path,
        dest="solution_path",
        help="Custom solution CSV (overrides --split)",
    )
    parser.add_argument(
        "--row-id", "-r", default="query_id",
        help="Row ID column name (default: query_id)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show missed gold citations per query",
    )
    args = parser.parse_args()

    solution_path = args.solution_path or (DEFAULT_DATA_DIR / f"{args.split}.csv")

    if not args.logs.exists():
        print(f"Error: logs file not found: {args.logs}", file=sys.stderr)
        sys.exit(1)
    if not solution_path.exists():
        print(f"Error: solution file not found: {solution_path}", file=sys.stderr)
        sys.exit(1)

    # Load gold
    sol = pd.read_csv(solution_path)
    gold_col = "gold_citations" if "gold_citations" in sol.columns else sol.columns[-1]
    gold_by_id = {
        str(row[args.row_id]): _parse_gold(row[gold_col])
        for _, row in sol.iterrows()
    }

    # Load candidate pools
    pool_by_id: dict[str, Set[str]] = {}
    with open(args.logs, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            qid = str(entry.get("query_id", ""))
            if qid:
                pool_by_id[qid] = _pool_from_logs(entry)

    print(f"Logs: {args.logs} ({len(pool_by_id)} queries)")
    print(f"Solution: {solution_path} ({len(gold_by_id)} queries)")

    missing = set(gold_by_id) - set(pool_by_id)
    if missing:
        print(f"WARNING: {len(missing)} gold queries have no logs: {sorted(missing)[:5]}...")
    print()

    recalls: list[float] = []
    oracle_f1s: list[float] = []
    total_gold = 0
    total_found = 0
    total_pool = 0
    per_query: list[tuple[str, int, int, int, float]] = []

    for qid, gold in gold_by_id.items():
        pool = pool_by_id.get(qid, set())
        found = gold & pool
        n_gold, n_found, n_pool = len(gold), len(found), len(pool)

        recall = (n_found / n_gold) if n_gold else 1.0
        # Oracle: perfect selection -> predict exactly (gold ∩ pool). precision=1.
        # F1 = 2*r / (1 + r); equals 1.0 when gold is empty.
        oracle_f1 = (2 * recall / (1 + recall)) if n_gold else 1.0

        recalls.append(recall)
        oracle_f1s.append(oracle_f1)
        total_gold += n_gold
        total_found += n_found
        total_pool += n_pool
        per_query.append((qid, n_gold, n_pool, n_found, recall))

    n = len(recalls)
    macro_recall = sum(recalls) / n if n else 0.0
    micro_recall = (total_found / total_gold) if total_gold else 0.0
    oracle_macro_f1 = sum(oracle_f1s) / n if n else 0.0

    print("Per-query (gold | pool | found | recall):")
    for qid, n_gold, n_pool, n_found, recall in sorted(per_query):
        print(f"  {qid}: gold={n_gold:>3}  pool={n_pool:>4}  found={n_found:>3}  recall={recall:.3f}")

    print()
    print("=== Candidate-pool recall (before selection) ===")
    print(f"  macro recall@candidates : {macro_recall:.4f}")
    print(f"  micro recall@candidates : {micro_recall:.4f}  ({total_found}/{total_gold})")
    print(f"  oracle Macro F1 ceiling : {oracle_macro_f1:.4f}  (if selection were perfect)")
    print(f"  avg pool size           : {total_pool / n:.1f}" if n else "")

    if args.verbose:
        print("\n=== Missed gold (never retrieved into the pool) ===")
        for qid in sorted(gold_by_id):
            missed = gold_by_id[qid] - pool_by_id.get(qid, set())
            if missed:
                print(f"  {qid} ({len(missed)} missed):")
                for c in sorted(missed):
                    print(f"      {c}")


if __name__ == "__main__":
    main()
