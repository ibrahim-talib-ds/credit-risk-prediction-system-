import os, glob, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
MODEL_DIR = "models"
REPORT_DIR = "reports"
DATA_DIR = "data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

KAGGLE_KEEP = ["loan_amnt", "funded_amnt", "term", "int_rate", "installment", "grade", "emp_length", "home_ownership", "annual_inc", "verification_status", "purpose", "dti", "fico_range_low", "fico_range_high", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "loan_status"]

def find_csv():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    files = [f for f in files if os.path.getsize(f) > 1_000_000]
    if not files: raise FileNotFoundError(f"No CSV > 1MB found in {DATA_DIR}/.")
    files.sort(key=os.path.getsize, reverse=True)
    return files[0]

def load_data():
    path = find_csv()
    print(f"Loading {path}")
    head = pd.read_csv(path, nrows=5, low_memory=False)
    keep = [c for c in KAGGLE_KEEP if c in head.columns]
    if "Default" in head.columns: 
        df = pd.read_csv(path, low_memory=False)
    elif "loan_status" in head.columns: 
        df = pd.read_csv(path, usecols=keep, low_memory=False)
    else: 
        df = pd.read_csv(path, low_memory=False)
    print("Shape:", df.shape)
    return df

def basic_cleaning(df):
    if "Default" not in df.columns:
        keep_status = ["Fully Paid", "Charged Off", "Default"]
        df = df[df["loan_status"].isin(keep_status)].copy()
        df["Default"] = df["loan_status"].apply(lambda x: 0 if x == "Fully Paid" else 1)
        df.drop(columns=["loan_status"], inplace=True)
    print(df["Default"].value_counts(normalize=True).round(4))
    return df

def eda(df):
    plt.figure(figsize=(5, 3))
    df["Default"].value_counts().plot(kind="bar", color=["steelblue", "tomato"])
    plt.title("Default Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "class_balance.png"))
    plt.close()

    key_num = [c for c in ["fico_range_low", "dti", "annual_inc", "loan_amnt"] if c in df.columns]
    if key_num:
        fig, axes = plt.subplots(1, len(key_num), figsize=(5 * len(key_num), 4))
        if len(key_num) == 1: axes = [axes]
        for ax, col in zip(axes, key_num):
            sns.boxplot(data=df, x="Default", y=col, ax=ax)
            ax.set_title(f"{col} by Default")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, "target_vs_features.png"))
        plt.close()

def split_and_preprocess(df):
    X, y = df.drop(columns=["Default"]), df["Default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    for col in X_train.select_dtypes(include=[np.number]).columns:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

    for col in X_train.select_dtypes(include=["object"]).columns:
        mode_val = X_train[col].mode(dropna=True)
        fill_val = mode_val.iloc[0] if len(mode_val) else "missing"
        X_train[col] = X_train[col].fillna(fill_val)
        X_test[col] = X_test[col].fillna(fill_val)
        le = LabelEncoder()
        le.fit(pd.concat([X_train[col], X_test[col]]).astype(str).unique())
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        joblib.dump(le, os.path.join(MODEL_DIR, f"encoder_{col}.pkl"))

    if "loan_amnt" in X_train.columns and "annual_inc" in X_train.columns:
        X_train["loan_to_income"] = X_train["loan_amnt"] / (X_train["annual_inc"] + 1)
        X_test["loan_to_income"] = X_test["loan_amnt"] / (X_test["annual_inc"] + 1)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train, X_test, y_test):
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE)
    }
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {"model": model, "auc": roc_auc_score(y_test, y_prob), "f1": f1_score(y_test, (y_prob >= 0.5).astype(int))}
        print(f"AUC: {results[name]['auc']:.4f} F1: {results[name]['f1']:.4f}")
    return results, spw

def tune_xgboost(X_train, y_train, spw):
    param_dist = {"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1, 0.15], "n_estimators": [100, 200], "subsample": [0.7, 0.9]}
    base = XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(base, param_dist, n_iter=5, scoring="roc_auc", cv=3, n_jobs=-1, random_state=RANDOM_STATE)
    rs.fit(X_train, y_train)
    return rs.best_estimator_

def evaluate_final(model, X_train, y_train, X_test, y_test, name, tag):
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.2, 0.8, 0.05)
    f1s = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])
    joblib.dump(model, os.path.join(MODEL_DIR, f"{tag}.pkl"))
    joblib.dump(best_t, os.path.join(MODEL_DIR, f"{tag}_threshold.pkl"))

    y_pred = (y_prob >= best_t).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Fully Paid", "Default"])
    disp.plot(cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(os.path.join(REPORT_DIR, f"{tag}_confusion_matrix.png"))
    plt.close()

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{name} ROC Curve")
    plt.savefig(os.path.join(REPORT_DIR, f"{tag}_roc_curve.png"))
    plt.close()

    try:
        imp = pd.Series(model.feature_importances_, index=X_test.columns)
        imp.sort_values(ascending=False).head(15).plot(kind="barh")
        plt.title("Top 15 Feature Importances")
        plt.savefig(os.path.join(REPORT_DIR, f"{tag}_feature_importance.png"))
        plt.close()
    except Exception: 
        pass

    with open(os.path.join(REPORT_DIR, "summary.txt"), "w") as f:
        f.write(f"Final model: {name}\nTest AUC: {roc_auc_score(y_test, y_prob):.4f}\nThreshold: {best_t:.2f}\n")
    return {"name": name}
