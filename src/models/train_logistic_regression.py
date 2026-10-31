import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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
    build_preprocessor,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


# Project paths
DATA_DIR = Path("data/processed/splits")
MODEL_DIR = Path("models")
METRICS_DIR = Path("reports/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# 1. Load training and validation data
print("Loading datasets...")

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X_train = train[FEATURES]
y_train = train["target"].astype(int)

X_validation = validation[FEATURES]
y_validation = validation["target"].astype(int)


# 2. Build complete machine learning pipeline
model = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=1.0,
                random_state=42,
            ),
        ),
    ]
)


# 3. Train the model
print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

print("Training completed.")


# 4. Predict on validation data
print("\nEvaluating on validation data...")

y_pred = model.predict(X_validation)
y_probability = model.predict_proba(X_validation)[:, 1]


# 5. Calculate evaluation metrics
metrics = {
    "model": "Logistic Regression",
    "accuracy": float(accuracy_score(y_validation, y_pred)),
    "precision": float(precision_score(y_validation, y_pred)),
    "recall": float(recall_score(y_validation, y_pred)),
    "f1_score": float(f1_score(y_validation, y_pred)),
    "roc_auc": float(roc_auc_score(y_validation, y_probability)),
    "pr_auc": float(average_precision_score(y_validation, y_probability)),
}

matrix = confusion_matrix(y_validation, y_pred)

print("\n" + "=" * 55)
print("LOGISTIC REGRESSION - VALIDATION RESULTS")
print("=" * 55)

for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

print("\nConfusion Matrix:")
print(matrix)

print("\nClassification Report:")
report = classification_report(
    y_validation,
    y_pred,
    target_names=["Fully Paid (0)", "Default (1)"],
)
print(report)


# 6. Save results
with open(METRICS_DIR / "logistic_regression_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

with open(METRICS_DIR / "logistic_regression_report.txt", "w") as file:
    file.write(report)

pd.DataFrame(
    matrix,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"],
).to_csv(METRICS_DIR / "logistic_regression_confusion_matrix.csv")

pd.DataFrame({
    "actual": y_validation,
    "predicted": y_pred,
    "default_probability": y_probability,
}).to_csv(
    METRICS_DIR / "logistic_regression_validation_predictions.csv",
    index=False,
)

joblib.dump(
    model,
    MODEL_DIR / "logistic_regression_baseline.joblib",
)

print("\nSaved model and evaluation reports.")
print("LOGISTIC REGRESSION BASELINE COMPLETED.")
