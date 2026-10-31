import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)

DATA_DIR = Path("data/processed/splits")
MODEL_DIR = Path("models")

validation = pd.read_csv(DATA_DIR / "validation.csv")

from src.features.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = validation[features]
y = validation["target"].astype(int)

thresholds = np.arange(0.10, 0.81, 0.025)

all_results = []

for model_name in [
    "catboost_baseline",
    "xgboost_baseline",
]:

    print(f"\n{'=' * 65}")
    print(model_name.upper())
    print("=" * 65)

    model = joblib.load(MODEL_DIR / f"{model_name}.joblib")

    probabilities = model.predict_proba(X)[:, 1]

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        row = {
            "model": model_name,
            "threshold": round(float(threshold), 3),
            "precision": precision_score(
                y, predictions, zero_division=0
            ),
            "recall": recall_score(
                y, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y, predictions, zero_division=0
            ),
            "f2_score": fbeta_score(
                y, predictions, beta=2, zero_division=0
            ),
        }

        results.append(row)
        all_results.append(row)

    results_df = pd.DataFrame(results)

    print("\nTOP 5 THRESHOLDS BY F1:")
    print(
        results_df.sort_values(
            "f1_score", ascending=False
        ).head(5).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nTOP 5 THRESHOLDS BY F2:")
    print(
        results_df.sort_values(
            "f2_score", ascending=False
        ).head(5).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

output = pd.DataFrame(all_results)

output.to_csv(
    Path("reports/metrics/threshold_analysis.csv"),
    index=False,
)

print("\nSaved: reports/metrics/threshold_analysis.csv")
print("THRESHOLD ANALYSIS COMPLETED.")
