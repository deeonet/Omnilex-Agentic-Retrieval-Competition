"""Fine-tune the multilingual embedder with MNRL (LQ-RAG FT Layer, Algorithm 1).

Loss: MultipleNegativesRankingLoss (the paper's MNRL) over (query, gold-citation-text)
pairs, with in-batch negatives. Applies the e5 ``query:`` / ``passage:`` prefixes so
inference (corpus.DenseIndex) matches training exactly.

Usage:
    python scripts/finetune_embedding.py \
        --base models/multilingual-e5-large \
        --out  models/e5-law-embed \
        --epochs 3 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FT = REPO / "data" / "processed" / "embed_ft"


def prefix(text: str, is_query: bool) -> str:
    return ("query: " if is_query else "passage: ") + text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(REPO / "models" / "multilingual-e5-large"))
    ap.add_argument("--out", default=str(REPO / "models" / "e5-law-embed"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base embedder {args.base} on {device}")
    model = SentenceTransformer(args.base, device=device)

    # --- training pairs ---
    examples: list[InputExample] = []
    with open(FT / "train_pairs.jsonl") as f:
        for line in f:
            p = json.loads(line)
            examples.append(InputExample(texts=[prefix(p["query"], True), prefix(p["positive"], False)]))
    print(f"{len(examples)} training pairs")

    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    # --- IR evaluator (prefixed) ---
    evaluator = None
    corpus_p = FT / "eval_corpus.json"
    if corpus_p.exists():
        corpus = {k: prefix(v, False) for k, v in json.load(open(corpus_p)).items()}
        queries = {k: prefix(v, True) for k, v in json.load(open(FT / "eval_queries.json")).items()}
        qrels = {k: set(v) for k, v in json.load(open(FT / "eval_qrels.json")).items()}
        queries = {k: q for k, q in queries.items() if qrels.get(k)}
        evaluator = InformationRetrievalEvaluator(
            queries=queries, corpus=corpus, relevant_docs=qrels,
            mrr_at_k=[10], recall_at_k=[5, 10, 20], name="law-ir",
            show_progress_bar=True,
        )
        print(f"IR eval: {len(queries)} queries, {len(corpus)} corpus docs")
        print("Baseline (before fine-tune):")
        evaluator(model, output_path=None)

    warmup = int(len(loader) * args.epochs * args.warmup_frac)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"Training {args.epochs} epochs, warmup={warmup} steps → {args.out}")
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        evaluator=evaluator,
        evaluation_steps=max(len(loader) // 2, 1) if evaluator else 0,
        output_path=args.out,
        save_best_model=True if evaluator else False,
        show_progress_bar=True,
        use_amp=True,
    )
    model.save(args.out)
    print(f"Saved fine-tuned embedder to {args.out}")


if __name__ == "__main__":
    main()
