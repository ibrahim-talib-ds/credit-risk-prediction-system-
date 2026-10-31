# Credit Risk Prediction Using Machine Learning

**Author:** Ibrahim Talib  
**Domain:** Financial Risk Analysis  
**Project Type:** Data Science / Machine Learning  
**Language:** Python

---

## 1. Project Overview

This project develops a machine learning system to estimate the
probability that a LendingClub loan will default.

It analyzes historical loan data, identifies credit-risk patterns,
trains classification models, and evaluates their predictive performance.

The modeling scope is limited to 36-month loans.

## 2. Problem Definition

Loan defaults can cause significant financial losses for lenders.

This project treats credit-risk prediction as a binary classification task.

- **0 — Fully Paid**
- **1 — Default**

The system estimates default risk for loans accepted by LendingClub.
It is not a loan-approval system.

## 3. Dataset

**Source:** LendingClub accepted loan dataset.

**Original file:** `accepted_2007_to_2018Q4.csv`

- Original records: 1,775,989
- Prepared records: 639,997
- Selected 36-month loans: 604,688
- Default rate in the selected population: 14.78%

Unresolved loan outcomes were excluded from the modeling dataset.

## 4. Machine Learning Workflow

The project follows a structured machine learning workflow:

1. Problem definition
2. Dataset loading and inspection
3. Exploratory Data Analysis (EDA)
4. Data cleaning and preprocessing
5. Feature engineering and selection
6. Chronological train, validation, and test splitting
7. Baseline model training
8. Model comparison and evaluation
9. Hyperparameter tuning and threshold analysis
10. Final evaluation on the untouched test set

## 5. Data Splitting

A chronological split was used to evaluate performance on later loans.

| Dataset | Issue year | Records |
|---|---:|---:|
| Training | Up to 2014 | 212,425 |
| Validation | 2015 | 283,026 |
| Testing | 2016 | 109,237 |

The 2016 test dataset was kept separate until final evaluation.

## 6. Models Evaluated

Five classification algorithms were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- CatBoost

Additional experiments included hyperparameter tuning,
feature ablation, feature importance, and classification
threshold analysis.

## 7. Selected Model

**Selected model:** CatBoost

The baseline CatBoost model achieved the strongest overall
validation results among the five baseline models.

The selected classification threshold was 0.525,
based on the best F1-score within the tested threshold grid.

## 8. Final Test Results

The selected model was evaluated on the untouched 2016 test dataset.

| Metric | Result |
|---|---:|
| Accuracy | 69.29% |
| Precision | 29.93% |
| Recall | 57.96% |
| F1-score | 39.48% |
| ROC-AUC | 71.19% |
| PR-AUC | 33.23% |

These results indicate that the model identifies a proportion
of default cases, but its predictions still have limitations.

## 9. Project Structure

- `data/` — Original and processed datasets
- `src/data/` — Dataset preparation and splitting
- `src/features/` — Preprocessing pipeline
- `src/models/` — Model training and evaluation
- `src/visualization/` — Exploratory data analysis
- `models/` — Saved trained models
- `reports/figures/` — EDA visualizations
- `reports/metrics/` — Evaluation metrics and experiment results
- `reports/final/` — Final project reports
- `notebooks/` — Jupyter notebooks
- `tests/` — Project tests

## 10. Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- CatBoost
- Matplotlib and Seaborn
- JupyterLab
- Joblib

## 11. Installation

Create and activate a virtual environment:

    python3 -m venv venv
    source venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

## 12. Important Project Files

- `src/data/prepare_dataset.py`
- `src/data/split_dataset.py`
- `src/visualization/eda.py`
- `src/features/preprocessing.py`
- `src/models/compare_models.py`
- `src/models/tune_models.py`
- `src/models/final_test_evaluation.py`

Final reports are available in `reports/final/`.

## 13. Limitations

- The dataset contains historical LendingClub loans.
- Only loans with resolved outcomes were included.
- The model is limited to 36-month loans.
- Predictive performance does not guarantee future results.
- The model is an educational machine learning project,
  not a production lending decision system.

---

**Project Status:** Core modeling and final test evaluation completed.

**Author:** Ibrahim Talib
