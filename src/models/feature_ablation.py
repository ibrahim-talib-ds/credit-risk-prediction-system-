import json
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from catboost import CatBoostClassifier

from src.features.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

DATA_DIR = Path("data/processed/splits")
OUTPUT_DIR = Path("reports/metrics")

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")

y_train = train["target"].astype(int)
y_val = validation["target"].astype(int)

experiments = {
    "full_features": {
        "remove": [],
    },
    "without_lender_pricing": {
        "remove": ["int_rate", "grade", "sub_grade"],
    },
    "without_state": {
        "remove": ["addr_state"],
    },
}

results = []

for experiment_name, config in experiments.items():

    removed = config["remove"]

    numeric = [
        feature for feature in NUMERIC_FEATURES
        if feature not in removed
    ]

    categorical = [
        feature for feature in CATEGORICAL_FEATURES
        if feature not in removed
    ]

    features = numeric + categorical

    print(f"\n{'=' * 60}")
    print("EXPERIMENT:", experiment_name)
    print("=" * 60)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown",
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
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
                verbose=False,
                allow_writing_files=False,
            ),
        ),
    ])

    X_train = train[features]
    X_val = validation[features]

    print("Training with", len(features), "original features...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "experiment": experiment_name,
        "removed_features": ", ".join(removed) if removed else "None",
        "feature_count": len(features),
        "accuracy": float((predictions == y_val).mean()),
        "precision": float(
            precision_score(y_val, predictions)
        ),
        "recall": float(
            recall_score(y_val, predictions)
        ),
        "f1_score": float(
            f1_score(y_val, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(y_val, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_val, probabilities)
        ),
    }

    results.append(metrics)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

comparison = pd.DataFrame(results)

comparison.to_csv(
    OUTPUT_DIR / "feature_ablation_comparison.csv",
    index=False,
)

print("\nFEATURE ABLATION COMPARISON")
print("=" * 75)

print(
    comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nSaved: reports/metrics/feature_ablation_comparison.csv")
print("FEATURE ABLATION COMPLETED.")
