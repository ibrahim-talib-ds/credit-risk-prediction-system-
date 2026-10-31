import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.features.preprocessing import (
    build_preprocessor,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

DATA_DIR = Path("data/processed/splits")
MODEL_DIR = Path("models/tuned")
METRICS_DIR = Path("reports/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

print("Loading training and validation data...")

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train[FEATURES]
y_train = train["target"].astype(int)

X_val = validation[FEATURES]
y_val = validation["target"].astype(int)

print("Fitting preprocessing pipeline on training data...")

preprocessor = build_preprocessor()

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

print("Training matrix:", X_train_processed.shape)
print("Validation matrix:", X_val_processed.shape)

negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())
class_ratio = negative / positive

experiments = [
    (
        "xgboost",
        "xgb_depth4",
        XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=10,
            reg_lambda=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=class_ratio,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=2,
            random_state=42,
        ),
    ),
    (
        "xgboost",
        "xgb_regularized",
        XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            min_child_weight=20,
            reg_lambda=10,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=class_ratio,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=2,
            random_state=42,
        ),
    ),
    (
        "xgboost",
        "xgb_depth6",
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=20,
            reg_lambda=10,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=class_ratio,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=2,
            random_state=42,
        ),
    ),
    (
        "catboost",
        "catboost_depth5",
        CatBoostClassifier(
            iterations=700,
            depth=5,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=42,
            thread_count=2,
            verbose=False,
            allow_writing_files=False,
        ),
    ),
    (
        "catboost",
        "catboost_depth6",
        CatBoostClassifier(
            iterations=700,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=42,
            thread_count=2,
            verbose=False,
            allow_writing_files=False,
        ),
    ),
    (
        "catboost",
        "catboost_depth7",
        CatBoostClassifier(
            iterations=500,
            depth=7,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=42,
            thread_count=2,
            verbose=False,
            allow_writing_files=False,
        ),
    ),
]

results = []

for family, name, classifier in experiments:

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {name}")
    print("=" * 60)

    classifier.fit(X_train_processed, y_train)

    prediction = classifier.predict(X_val_processed)
    prediction = prediction.astype(int).ravel()

    probability = classifier.predict_proba(X_val_processed)[:, 1]

    metrics = {
        "model": name,
        "family": family,
        "accuracy": float(accuracy_score(y_val, prediction)),
        "precision": float(precision_score(y_val, prediction)),
        "recall": float(recall_score(y_val, prediction)),
        "f1_score": float(f1_score(y_val, prediction)),
        "roc_auc": float(roc_auc_score(y_val, probability)),
        "pr_auc": float(average_precision_score(y_val, probability)),
    }

    results.append(metrics)

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier),
        ]
    )

    # Reuse the already-fitted preprocessor.
    full_pipeline.steps[0] = ("preprocessor", preprocessor)

    joblib.dump(
        full_pipeline,
        MODEL_DIR / f"{name}.joblib",
    )

comparison = pd.DataFrame(results)
comparison = comparison.sort_values(
    by="pr_auc",
    ascending=False,
)

comparison.to_csv(
    METRICS_DIR / "tuning_comparison.csv",
    index=False,
)

with open(METRICS_DIR / "tuning_comparison.json", "w") as file:
    json.dump(results, file, indent=4)

print("\n" + "=" * 75)
print("TUNING EXPERIMENT LEADERBOARD")
print("=" * 75)

print(
    comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nSaved tuned candidate models in:", MODEL_DIR)
print("Saved comparison in:", METRICS_DIR / "tuning_comparison.csv")
print("\nMODEL TUNING EXPERIMENT COMPLETED.")
