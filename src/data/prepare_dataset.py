from pathlib import Path
import pandas as pd

RAW_FILE = Path("data/accepted_2007_to_2018Q4.csv")
OUTPUT_FILE = Path("data/processed/credit_risk_modeling.csv")

TARGET_MAP = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

FEATURES = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
]

USECOLS = FEATURES + ["issue_d", "loan_status"]

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    total_kept = 0
    total_defaults = 0
    first_chunk = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            RAW_FILE,
            usecols=USECOLS,
            chunksize=50_000,
            low_memory=False,
        ),
        start=1,
    ):
        chunk["target"] = chunk["loan_status"].map(TARGET_MAP)

        chunk = chunk.dropna(
            subset=["target", "issue_d", "term"]
        )

        chunk["issue_date"] = pd.to_datetime(
            chunk["issue_d"],
            format="%b-%Y",
            errors="coerce",
        )

        chunk["term_months"] = pd.to_numeric(
            chunk["term"].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce",
        )

        chunk = chunk.dropna(
            subset=["issue_date", "term_months"]
        )

        issue_index = (
            chunk["issue_date"].dt.year * 12
            + chunk["issue_date"].dt.month
        )

        cutoff_index = 2019 * 12 + 4

        mature = (
            issue_index + chunk["term_months"]
            <= cutoff_index
        )

        chunk = chunk.loc[mature].copy()

        if chunk.empty:
            continue

        chunk["issue_year"] = chunk["issue_date"].dt.year

        chunk = chunk[
            FEATURES + ["issue_year", "target"]
        ]

        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        first_chunk = False
        total_kept += len(chunk)
        total_defaults += int(chunk["target"].sum())

        print(
            f"Chunk {chunk_number}: "
            f"kept {len(chunk):,} | "
            f"total {total_kept:,}"
        )

    print("\nDATASET PREPARATION COMPLETED")
    print(f"Total loans: {total_kept:,}")
    print(f"Defaults: {total_defaults:,}")

    if total_kept:
        print(
            f"Default rate: "
            f"{total_defaults / total_kept * 100:.2f}%"
        )
        print(f"Saved to: {OUTPUT_FILE}")
    else:
        print("No records were saved. Check the input data.")

if __name__ == "__main__":
    main()
