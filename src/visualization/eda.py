from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = Path("data/processed/credit_risk_modeling.csv")
FIGURES_DIR = Path("reports/figures")
METRICS_DIR = Path("reports/metrics")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    print("Loading modeling dataset...")
    df = pd.read_csv(DATA_FILE)

    print("\n" + "=" * 65)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 65)

    # 1. Dataset overview
    print("\n1. DATASET OVERVIEW")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nData types:")
    print(df.dtypes.to_string())

    # 2. Statistical summary
    print("\n2. NUMERICAL STATISTICS")
    numeric = df.select_dtypes(include="number")
    statistics = numeric.describe().T
    statistics.to_csv(METRICS_DIR / "eda_statistics.csv")
    print(statistics.to_string())

    # 3. Missing values
    print("\n3. MISSING VALUES")
    missing = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": df.isnull().mean() * 100
    })
    missing = missing[missing["missing_count"] > 0]
    missing = missing.sort_values("missing_percent", ascending=False)
    missing.to_csv(METRICS_DIR / "eda_missing_values.csv")
    print(missing.to_string())

    # 4. Duplicate records
    print("\n4. DUPLICATE RECORDS")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    # 5. Target distribution
    print("\n5. TARGET DISTRIBUTION")
    target_counts = df["target"].value_counts().sort_index()
    print(target_counts)
    print("\nTarget percentages:")
    print((df["target"].value_counts(normalize=True) * 100).round(2))

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="target")
    plt.title("Loan Outcome Distribution")
    plt.xlabel("Target (0 = Fully Paid, 1 = Default)")
    plt.ylabel("Number of Loans")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close()

    # 6. Loans by issue year
    print("\n6. LOANS BY ISSUE YEAR")
    yearly = df.groupby("issue_year").agg(
        total_loans=("target", "size"),
        defaults=("target", "sum"),
        default_rate=("target", "mean")
    )
    yearly["default_rate"] *= 100
    yearly.to_csv(METRICS_DIR / "eda_yearly_analysis.csv")
    print(yearly.round(2).to_string())

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="issue_year", y="target", errorbar=None)
    plt.title("Default Rate by Loan Issue Year")
    plt.xlabel("Issue Year")
    plt.ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "default_rate_by_year.png", dpi=150)
    plt.close()

    # 7. Categorical feature relationships
    print("\n7. DEFAULT RATE BY IMPORTANT CATEGORIES")

    for column in ["grade", "term", "home_ownership", "purpose"]:
        table = df.groupby(column, dropna=False)["target"].agg(
            total_loans="count",
            defaults="sum",
            default_rate="mean"
        )
        table["default_rate"] *= 100
        table = table.sort_values("default_rate", ascending=False)
        table.to_csv(METRICS_DIR / f"default_rate_by_{column}.csv")
        print(f"\n{column.upper()}:")
        print(table.round(2).to_string())

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="grade", y="target", order=sorted(df["grade"].dropna().unique()), errorbar=None)
    plt.title("Default Rate by LendingClub Grade")
    plt.xlabel("Loan Grade")
    plt.ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "default_rate_by_grade.png", dpi=150)
    plt.close()

    # 8. Outlier analysis using IQR
    print("\n8. NUMERICAL OUTLIER ANALYSIS (IQR METHOD)")

    outlier_results = []

    for column in numeric.columns:
        if column in ["target", "issue_year"]:
            continue

        values = df[column].dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = ((values < lower) | (values > upper)).sum()

        outlier_results.append({
            "feature": column,
            "outlier_count": int(count),
            "outlier_percent": count / len(values) * 100,
            "lower_bound": lower,
            "upper_bound": upper
        })

    outliers = pd.DataFrame(outlier_results)
    outliers.to_csv(METRICS_DIR / "eda_outlier_analysis.csv", index=False)
    print(outliers.round(2).to_string(index=False))

    # 9. Feature distributions
    print("\n9. FEATURE DISTRIBUTION CHARTS")

    for column in ["loan_amnt", "int_rate", "annual_inc", "fico_range_low"]:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x=column, bins=50, kde=True)
        plt.title(f"Distribution of {column}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"distribution_{column}.png", dpi=150)
        plt.close()

    # 10. Correlation analysis
    print("\n10. CORRELATION ANALYSIS")

    correlation = numeric.corr()
    correlation.to_csv(METRICS_DIR / "eda_correlation_matrix.csv")

    plt.figure(figsize=(15, 12))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        annot=False
    )
    plt.title("Numerical Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    print("\nEDA COMPLETED SUCCESSFULLY.")
    print(f"Charts saved in: {FIGURES_DIR}")
    print(f"Analysis tables saved in: {METRICS_DIR}")


if __name__ == "__main__":
    main()
