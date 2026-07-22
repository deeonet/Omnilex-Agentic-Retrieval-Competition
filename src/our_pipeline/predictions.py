import json
from pathlib import Path

from tqdm import tqdm
import pandas as pd
from llm.define_agent import run_agent

# Hard limit enforced by the submission validator (scripts/evaluate_submission.py).
MAX_FIELD_CHARS = 10_000


def generate_predictions(
    test_df: pd.DataFrame, TOOLS, sibling_index: dict | None = None, eval_agent=None
) -> tuple[pd.DataFrame, list[dict]]:
    """Run the retrieval agent over every query.

    Args:
        sibling_index: optional decision -> considerations map passed through to
            ``run_agent`` for court sibling-consideration expansion.
        eval_agent: optional ``EvaluationAgent``. When provided, its groundedness/
            relevance filter is applied to the recall-maximised candidate pool to
            drop false positives (the Macro-F1 precision lever).

    Returns:
        Tuple of (predictions_df, all_logs). ``all_logs`` holds one entry per
        query with the decomposition and per-search details for debugging
        retrieval recall.
    """
    predictions = []
    all_logs = []  # Store logs for all queries

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running agent"):
        query_id = row["query_id"]
        query_text = row["query"]

        # Run agent (recall-maximising candidate pool)
        raw_citations, logs = run_agent(
            query_text, tools=TOOLS, verbose=False, sibling_index=sibling_index
        )

        # Evaluation agent: precision filter over the candidate pool
        final_citations = raw_citations
        if eval_agent is not None:
            final_citations, eval_log = eval_agent.filter(query_text, raw_citations)
            logs.append(eval_log)

        # Submission validator rejects predicted_citations > 10,000 chars.
        # Candidates are recall-ranked, so drop from the tail until we fit.
        n_before = len(final_citations)
        while final_citations and len(";".join(final_citations)) > MAX_FIELD_CHARS:
            final_citations.pop()
        if len(final_citations) < n_before:
            logs.append({
                "step": "length_cap",
                "dropped": n_before - len(final_citations),
                "kept": len(final_citations),
            })

        # Store logs with query_id
        all_logs.append({
            "query_id": query_id,
            "query": query_text,
            "logs": logs,
        })

        predictions.append({
            "query_id": query_id,
            "predicted_citations": ";".join(final_citations),
        })

    print(f"\nGenerated predictions for {len(predictions)} queries")
    print(f"Collected logs for {len(all_logs)} queries")

    predictions_df = pd.DataFrame(predictions)
    return predictions_df, all_logs


def save_run_logs(all_logs: list[dict], path: Path | str) -> None:
    """Persist per-query retrieval logs as JSONL (one query per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in all_logs:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    print(f"Run logs saved to: {path}")