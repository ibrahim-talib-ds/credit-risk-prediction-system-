import joblib
import pandas as pd

from pathlib import Path

MODEL_PATH = Path("models/catboost_baseline.joblib")
OUTPUT_PATH = Path("reports/metrics/catboost_feature_importance.csv")

print("Loading CatBoost model...")

model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()
importance_values = classifier.feature_importances_

importance = pd.DataFrame({
    "encoded_feature": feature_names,
    "importance": importance_values,
})

# Group one-hot encoded columns under their original feature.
importance["original_feature"] = (
    importance["encoded_feature"]
    .str.replace(r"^(numeric|categorical)__", "", regex=True)
    .str.split("_")
    .str[0]
)

# Use the original feature names to group categorical columns correctly.
from src.features.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def find_original_feature(encoded_name):
    clean_name = encoded_name.split("__", 1)[-1]

    matches = [
        feature for feature in all_features
        if clean_name == feature or clean_name.startswith(feature + "_")
    ]

    return max(matches, key=len) if matches else clean_name

importance["original_feature"] = (
    importance["encoded_feature"].apply(find_original_feature)
)

grouped = (
    importance.groupby("original_feature")["importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

grouped["importance_percentage"] = (
    grouped["importance"] / grouped["importance"].sum() * 100
)

grouped.to_csv(OUTPUT_PATH, index=False)

print("\nTOP 20 ORIGINAL FEATURES")
print("=" * 55)

print(
    grouped.head(20).to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}"
    )
)

print(f"\nSaved: {OUTPUT_PATH}")
print("\nFEATURE IMPORTANCE ANALYSIS COMPLETED.")
