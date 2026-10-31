import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
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

DATA_DIR = Path("data/processed/splits")
MODEL_DIR = Path("models")
METRICS_DIR = Path("reports/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

print("Loading datasets...")

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train[FEATURES]
y_train = train["target"].astype(int)

X_validation = validation[FEATURES]
y_validation = validation["target"].astype(int)

model = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        (
            "classifier",
            CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                auto_class_weights="Balanced",
                random_seed=42,
                thread_count=2,
                verbose=100,
                allow_writing_files=False,
            ),
        ),
    ]
)

print("\nTraining CatBoost...")
model.fit(X_train, y_train)

print("Training completed.")

print("\nEvaluating on validation data...")

y_pred = model.predict(X_validation).astype(int).ravel()
y_probability = model.predict_proba(X_validation)[:, 1]

metrics = {
    "model": "CatBoost",
    "accuracy": float(accuracy_score(y_validation, y_pred)),
    "precision": float(precision_score(y_validation, y_pred)),
    "recall": float(recall_score(y_validation, y_pred)),
    "f1_score": float(f1_score(y_validation, y_pred)),
    "roc_auc": float(roc_auc_score(y_validation, y_probability)),
    "pr_auc": float(average_precision_score(y_validation, y_probability)),
}

matrix = confusion_matrix(y_validation, y_pred)

print("\n" + "=" * 55)
print("CATBOOST - VALIDATION RESULTS")
print("=" * 55)

for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

print("\nConfusion Matrix:")
print(matrix)

report = classification_report(
    y_validation,
    y_pred,
    target_names=["Fully Paid (0)", "Default (1)"],
)

print("\nClassification Report:")
print(report)

with open(METRICS_DIR / "catboost_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

with open(METRICS_DIR / "catboost_report.txt", "w") as file:
    file.write(report)

pd.DataFrame(
    matrix,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"],
).to_csv(METRICS_DIR / "catboost_confusion_matrix.csv")

joblib.dump(
    model,
    MODEL_DIR / "catboost_baseline.joblib",
)

print("\nSaved model and evaluation reports.")
print("CATBOOST BASELINE COMPLETED.")
