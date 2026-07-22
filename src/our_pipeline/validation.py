from omnilex.evaluation.scorer import Scorer, validate_submission_format
import pandas as pd

def validate_and_score_submission(query_file, submission_path):
    test_df = pd.read_csv(query_file)
    print("\nValidating submission:")
    validation_errors = validate_submission_format(submission_path)
    if validation_errors:
        print("Validation failed:")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("Validation passed")

    if "gold_citations" in test_df.columns:
        print(f"\nScoring submission against: {query_file}")
        scores = Scorer().score(submission_path, query_file)

        print("\nScores:")
        for metric, value in scores.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    else:
        print("\nNo gold_citations column found, so scoring is skipped for this file.")