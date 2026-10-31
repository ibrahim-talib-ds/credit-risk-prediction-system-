import json
import pandas as pd
from pathlib import Path

METRICS_DIR = Path("reports/metrics")

model_names = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "catboost",
]

results = []

for name in model_names:
    path = METRICS_DIR / f"{name}_metrics.json"

    with open(path) as file:
        metrics = json.load(file)

    results.append(metrics)

comparison = pd.DataFrame(results)

comparison = comparison[
    [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
    ]
]

comparison = comparison.sort_values(
    by="pr_auc",
    ascending=False,
)

output = METRICS_DIR / "model_comparison.csv"
comparison.to_csv(output, index=False)

print("\nBASELINE MODEL COMPARISON")
print("=" * 85)
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print(f"\nSaved report: {output}")
print("\nMODEL COMPARISON COMPLETED.")
