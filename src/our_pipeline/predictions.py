from tqdm import tqdm
import pandas as pd
from our_pipeline.llm.define_agent import run_agent
from our_pipeline.constants import CONFIG
from our_pipeline.search_tools import retrieve_union
from our_pipeline.rerank import get_reranker

def generate_predictions(test_df: pd.DataFrame, TOOLS, text_lookup=None) -> pd.DataFrame:
    predictions = []
    all_logs = []  # Store logs for all queries

    mode = CONFIG.get("retrieval_mode", "agent")
    desc = "Retrieving (union)" if mode == "recall_union" else "Running agent"

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=desc):
        query_id = row["query_id"]
        query_text = row["query"]

        # Guard per-query so a transient failure (e.g. an API 500 that survives client
        # retries, or a tool error) yields empty predictions for this query instead of
        # aborting the whole batch.
        try:
            if mode == "recall_union":
                # Recall-first: directly fuse every tool's hits (no LLM curation step).
                raw_citations = retrieve_union(
                    query_text,
                    TOOLS,
                    top_k=CONFIG.get("max_predictions"),
                    rrf_k=CONFIG.get("rrf_k", 60),
                )
                logs = [{"type": "recall_union", "n_citations": len(raw_citations)}]
            else:
                raw_citations, logs = run_agent(query_text, tools=TOOLS, verbose=False)
        except Exception as exc:  # noqa: BLE001 - one failed query must not kill the run
            print(f"  ! prediction failed for {query_id}: {exc}")
            raw_citations, logs = [], [{"type": "error", "error": str(exc)}]

        # Precision filter: keep only citations the reranker is confident about. Applied
        # after fusion/agent so it benefits both modes; a reranker failure leaves the
        # un-reranked citations untouched rather than dropping the query.
        if CONFIG.get("enable_reranking") and text_lookup is not None and raw_citations:
            n_before = len(raw_citations)
            try:
                raw_citations = get_reranker().rerank(query_text, raw_citations, text_lookup)
                logs.append({"type": "rerank", "n_before": n_before, "n_after": len(raw_citations)})
            except Exception as exc:  # noqa: BLE001 - reranker failure must not drop the query
                print(f"  ! rerank failed for {query_id}: {exc}")
                logs.append({"type": "rerank_error", "error": str(exc)})

        # Store logs with query_id
        all_logs.append({
            "query_id": query_id,
            "query": query_text,
            "logs": logs,
        })

        predictions.append({
            "query_id": query_id,
            "predicted_citations": ";".join(raw_citations),
        })

    print(f"\nGenerated predictions for {len(predictions)} queries")
    print(f"Collected logs for {len(all_logs)} queries")

    predictions_df = pd.DataFrame(predictions)
    return predictions_df