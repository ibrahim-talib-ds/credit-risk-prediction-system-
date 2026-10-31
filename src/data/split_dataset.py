import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/credit_risk_modeling.csv")
OUTPUT = Path("data/processed/splits")

print("Loading prepared dataset...")
df = pd.read_csv(INPUT)

print(f"Original records: {len(df):,}")

# Clean spaces in loan term values
df["term"] = df["term"].str.strip()

# First model scope: 36-month loans only
df = df[df["term"] == "36 months"].copy()

print(f"36-month loans selected: {len(df):,}")

# Data-quality corrections
df.loc[df["annual_inc"] == 0, "annual_inc"] = pd.NA
df.loc[(df["dti"] < 0) | (df["dti"] > 100), "dti"] = pd.NA

# Preserve missing delinquency information
df["mths_since_last_delinq_missing"] = (
    df["mths_since_last_delinq"].isna().astype(int)
)

# Approximate credit history length in years
earliest_year = pd.to_datetime(
    df["earliest_cr_line"],
    format="%b-%Y",
    errors="coerce"
).dt.year

df["credit_history_years"] = df["issue_year"] - earliest_year

df.loc[
    df["credit_history_years"] < 0,
    "credit_history_years"
] = pd.NA

df = df.drop(columns=["earliest_cr_line"])

# Chronological split
train = df[df["issue_year"] <= 2014].copy()
validation = df[df["issue_year"] == 2015].copy()
test = df[df["issue_year"] == 2016].copy()

OUTPUT.mkdir(parents=True, exist_ok=True)

train.to_csv(OUTPUT / "train.csv", index=False)
validation.to_csv(OUTPUT / "validation.csv", index=False)
test.to_csv(OUTPUT / "test.csv", index=False)

# Report
for name, data in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:
    print(f"\n{name}")
    print("-" * 45)
    print(f"Records: {len(data):,}")
    print(f"Defaults: {int(data['target'].sum()):,}")
    print(f"Default rate: {data['target'].mean() * 100:.2f}%")

print("\nSaved all three datasets successfully.")
print("CHRONOLOGICAL DATA SPLIT COMPLETED.")
