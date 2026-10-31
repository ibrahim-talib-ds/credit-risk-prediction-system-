# Final Model Selection — Credit Risk Prediction

## 1. Selected Model

CatBoost Classifier

Reason:
CatBoost achieved the strongest overall baseline validation results among the five models.

Validation performance:
- Accuracy: 65.03%
- Precision: 24.17%
- Recall: 63.16%
- F1-score: 34.97%
- ROC-AUC: 69.73%
- PR-AUC: 27.68%

## 2. Model Comparison

CatBoost was selected after comparing:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- CatBoost

CatBoost achieved the highest baseline F1-score, ROC-AUC, and PR-AUC, although its advantage over XGBoost was small.

## 3. Hyperparameter Tuning

Several XGBoost and CatBoost configurations were tested.

The tuning experiments did not produce a convincing improvement over the original CatBoost baseline.

Therefore, the original CatBoost configuration is retained.

## 4. Feature Ablation

Three feature experiments were conducted:

- Full features: ROC-AUC 69.73%, PR-AUC 27.68%
- Without lender pricing features: ROC-AUC 68.22%, PR-AUC 26.26%
- Without state: ROC-AUC 69.59%, PR-AUC 27.53%

The full feature set performed best. All 27 original features are retained.

## 5. Class Imbalance

The training data contains fewer default cases than fully paid cases.

Class imbalance was addressed using CatBoost's automatic balanced class weights.

## 6. Decision Threshold

The default threshold of 0.50 was compared with alternative thresholds.

The threshold 0.525 achieved the highest F1-score in the tested threshold grid:

- Precision: 25.27%
- Recall: 57.57%
- F1-score: 35.12%

Therefore, 0.525 is selected as the primary operating threshold because this project uses F1-score as its selection objective.

A threshold of 0.375 is also documented as a higher-recall alternative.

## 7. Final Decision

Selected model: CatBoost Classifier

Selected feature set: All 27 original features

Selected primary threshold: 0.525

The model and threshold decisions are based on training and validation results only.

## 8. Final Test Data Protection

The 2016 test dataset has not been used for model selection, tuning, feature selection, or threshold selection.

It will be used only for the final evaluation after the model decisions are frozen.
