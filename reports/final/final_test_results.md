# Final Test Evaluation Report — Credit Risk Prediction

## 1. Evaluation Setup

- Selected model: CatBoost Classifier
- Test period: 2016
- Test records: 109,237
- Actual defaults: 18,878
- Actual fully paid loans: 90,359
- Test default rate: 17.28%
- Decision threshold: 0.525

The model was loaded from its saved training artifact. The 2016 test dataset was used only for final evaluation, not model selection or tuning.

## 2. Final Test Metrics

| Metric | Result |
|---|---:|
| Accuracy | 69.29% |
| Precision | 29.93% |
| Recall | 57.96% |
| F1-score | 39.48% |
| ROC-AUC | 71.19% |
| PR-AUC | 33.23% |

## 3. Confusion Matrix

| Actual Outcome | Predicted Paid | Predicted Default |
|---|---:|---:|
| Fully Paid | 64,746 | 25,613 |
| Default | 7,937 | 10,941 |

The model correctly identified 10,941 default cases and missed 7,937 actual defaults.

## 4. Validation and Test Comparison

| Metric | Validation (2015) | Test (2016) |
|---|---:|---:|
| Accuracy | 65.03% | 69.29% |
| Precision | 24.17% | 29.93% |
| Recall | 63.16% | 57.96% |
| F1-score | 34.97% | 39.48% |
| ROC-AUC | 69.73% | 71.19% |
| PR-AUC | 27.68% | 33.23% |

The final test results do not show a performance collapse. The test default rate (17.28%) is higher than the validation default rate (14.89%), so the two periods have different class distributions.

## 5. Interpretation and Limitations

- ROC-AUC of 71.19% indicates useful ability to distinguish between paid and default loans.
- Recall of 57.96% means the model identified approximately 58% of actual defaults.
- Precision of 29.93% means that approximately 30% of loans predicted as defaults actually defaulted.
- False positives remain substantial: 25,613 fully paid loans were classified as defaults.
- The model should support risk assessment, not be treated as a perfect decision-maker.
- Results apply to the selected population of 36-month LendingClub loans and should not automatically be generalized to other loan terms or populations.

## 6. Final Conclusion

The CatBoost model was selected using training and validation experiments and evaluated on the untouched 2016 test dataset.

Final test ROC-AUC: 71.19%.

Final test PR-AUC: 33.23%.

Final test F1-score: 39.48%.

The final test evaluation is complete. Test results were not used to modify the selected model or decision threshold.
