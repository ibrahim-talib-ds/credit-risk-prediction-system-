import json
import joblib
from pathlib import Path

models = [
    "xgboost_baseline",
    "catboost_baseline",
    "logistic_regression_baseline",
]

for name in models:
    path = Path("models") / f"{name}.joblib"

    print(f"\n{'=' * 55}")
    print(name.upper())
    print("=" * 55)

    model = joblib.load(path)

    classifier = model.named_steps["classifier"]

    print("Classifier:", classifier.__class__.__name__)
    print("Parameters:")

    for key, value in classifier.get_params().items():
        if key in [
            "n_estimators",
            "iterations",
            "max_depth",
            "depth",
            "learning_rate",
            "min_child_weight",
            "reg_lambda",
            "scale_pos_weight",
            "auto_class_weights",
            "class_weight",
            "C",
        ]:
            print(f"  {key}: {value}")

print("\nBASELINE INSPECTION COMPLETED.")
