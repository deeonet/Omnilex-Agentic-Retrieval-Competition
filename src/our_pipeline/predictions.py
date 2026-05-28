from tqdm import tqdm
import pandas as pd
from our_pipeline.llm.define_agent import run_agent

def generate_predictions(test_df: pd.DataFrame, TOOLS) -> pd.DataFrame:
    predictions = []
    all_logs = []  # Store logs for all queries

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running agent"):
        query_id = row["query_id"]
        query_text = row["query"]

        # Run agent
        raw_citations, logs = run_agent(query_text, tools=TOOLS, verbose=False)

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