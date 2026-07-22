"""Build DQA + DInstr datasets for generative LoRA (LQ-RAG FT Layer, Algorithm 2).

Reframed for the competition (citation-set output, not prose):
  DQA   : query -> exact gold citation set (teaches decomposition + closed-vocab output)
  DInstr: statute/ruling text -> its canonical citation (teaches the Art./Abs./E. format)

Both are written as chat-format JSONL ({"messages": [...]}) for TRL SFTTrainer.
DInstr is held to a subset so DQA dominates, matching the paper's domain-heavy mix.
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
OUT = DATA / "processed" / "gen_ft"
csv.field_size_limit(10_000_000)

QA_SYSTEM = (
    "You are a Swiss legal citation retriever. Given a legal question, output the set "
    "of relevant citations — Swiss federal law articles and Federal Court decisions — "
    "exactly as they appear in the corpus, one per line. Output only citations."
)
INSTR_SYSTEM = (
    "You are a Swiss legal citation normalizer. Given the text of a statutory provision "
    "or a court consideration, output its exact canonical citation and nothing else."
)


def split_gold(cell: str) -> list[str]:
    return [c.strip() for c in str(cell).split(";") if c.strip()] if isinstance(cell, str) else []


def chat(system: str, user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instr-samples", type=int, default=4000)
    ap.add_argument("--instr-max-chars", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # --- DQA: query -> gold citation set ---
    tr = pd.read_csv(DATA / "train.csv").dropna(subset=["query", "gold_citations"])
    qa = []
    for _, row in tr.iterrows():
        golds = split_gold(row["gold_citations"])
        if golds:
            qa.append(chat(QA_SYSTEM, str(row["query"]).strip(), "\n".join(golds)))
    with open(OUT / "dqa.jsonl", "w") as f:
        for ex in qa:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"DQA: {len(qa)} examples")

    # --- DInstr: corpus text -> canonical citation (sampled from both corpora) ---
    instr = []
    laws = pd.read_csv(DATA / "laws_de.csv")
    law_rows = list(zip(laws["citation"], laws["text"]))
    rng.shuffle(law_rows)
    for cit, txt in law_rows[: args.instr_samples // 2]:
        instr.append(chat(INSTR_SYSTEM, str(txt)[: args.instr_max_chars], str(cit)))

    court_budget = args.instr_samples - len(instr)
    with open(DATA / "court_considerations.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, r in enumerate(reader):
            if i % 97 != 0:  # sparse stride so we sample across the 2.4M rows
                continue
            instr.append(chat(INSTR_SYSTEM, (r.get("text", "") or "")[: args.instr_max_chars], r.get("citation", "")))
            if len(instr) - (args.instr_samples // 2) >= court_budget:
                break
    rng.shuffle(instr)
    with open(OUT / "dinstr.jsonl", "w") as f:
        for ex in instr:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"DInstr: {len(instr)} examples")
    print(f"Wrote datasets to {OUT}")


if __name__ == "__main__":
    main()
