import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix

from src.features.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

validation = pd.read_csv(
    "data/processed/splits/validation.csv"
)

features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = validation[features]
y = validation["target"].astype(int)

operating_points = {
    "Default threshold": 0.50,
    "Best F1 threshold": 0.525,
    "Best F2 threshold": 0.375,
}

for model_name in ["catboost_baseline", "xgboost_baseline"]:

    model = joblib.load(f"models/{model_name}.joblib")
    probabilities = model.predict_proba(X)[:, 1]

    print(f"\n{'=' * 65}")
    print(model_name.upper())
    print("=" * 65)

    for label, threshold in operating_points.items():

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y, predictions
        ).ravel()

        print(f"\n{label}: {threshold}")
        print(f"True Negatives:  {tn:,}")
        print(f"False Positives: {fp:,}")
        print(f"False Negatives: {fn:,}")
        print(f"True Positives:  {tp:,}")

print("\nTHRESHOLD CONFUSION ANALYSIS COMPLETED.")
