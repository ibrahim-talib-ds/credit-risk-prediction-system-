import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from src.features.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

# Paths
MODEL_PATH = Path("models/catboost_baseline.joblib")
TEST_PATH = Path("data/processed/splits/test.csv")
OUTPUT_DIR = Path("reports/metrics")
FINAL_DIR = Path("reports/final")

# Previously selected on validation data
THRESHOLD = 0.525

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("FINAL TEST EVALUATION — CATBOOST")
print("=" * 60)

# Load the already-trained model and untouched test dataset
print("\nLoading saved CatBoost model...")
model = joblib.load(MODEL_PATH)

print("Loading 2016 test dataset...")
test = pd.read_csv(TEST_PATH)

features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X_test = test[features]
y_test = test["target"].astype(int)

print(f"Test records: {len(test):,}")
print(f"Test defaults: {int(y_test.sum()):,}")
print(f"Test default rate: {y_test.mean() * 100:.2f}%")
print(f"Selected threshold: {THRESHOLD}")

# Predict probabilities
print("\nGenerating predictions...")
probabilities = model.predict_proba(X_test)[:, 1]

# Apply the threshold selected using validation data
predictions = (probabilities >= THRESHOLD).astype(int)

# Calculate final test metrics
metrics = {
    "model": "CatBoost",
    "test_year": 2016,
    "threshold": THRESHOLD,
    "test_records": int(len(y_test)),
    "test_default_rate": float(y_test.mean()),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(y_test, predictions, zero_division=0)),
    "recall": float(recall_score(y_test, predictions, zero_division=0)),
    "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
    "pr_auc": float(average_precision_score(y_test, probabilities)),
}

# Confusion matrix
cm = confusion_matrix(y_test, predictions, labels=[0, 1])

confusion_df = pd.DataFrame(
    cm,
    index=["Actual_Paid", "Actual_Default"],
    columns=["Predicted_Paid", "Predicted_Default"],
)

confusion_df.to_csv(
    OUTPUT_DIR / "final_test_confusion_matrix.csv"
)

# Classification report
report = classification_report(
    y_test,
    predictions,
    target_names=["Fully Paid", "Default"],
    zero_division=0,
)

# Save metrics
with open(OUTPUT_DIR / "final_test_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

with open(OUTPUT_DIR / "final_test_classification_report.txt", "w") as file:
    file.write(report)

# Save test predictions for final analysis
prediction_df = pd.DataFrame({
    "actual": y_test,
    "default_probability": probabilities,
    "predicted": predictions,
})

prediction_df.to_csv(
    OUTPUT_DIR / "final_test_predictions.csv",
    index=False,
)

# Print results
print("\nFINAL TEST RESULTS")
print("=" * 60)

for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

print("\nCONFUSION MATRIX")
print(confusion_df)

print("\nCLASSIFICATION REPORT")
print(report)

print("\nSaved files:")
print("- reports/metrics/final_test_metrics.json")
print("- reports/metrics/final_test_confusion_matrix.csv")
print("- reports/metrics/final_test_classification_report.txt")
print("- reports/metrics/final_test_predictions.csv")

print("\nFINAL TEST EVALUATION COMPLETED.")
