# Machine Learning Metrics: Classification & Regression

| Type           | Metric           | Formula / Description                          | Best Value | When to Use / Notes                                           |
|----------------|------------------|-----------------------------------------------|------------|--------------------------------------------------------------|
| Classification | Accuracy         | (TP + TN) / (TP + TN + FP + FN)               | 1          | Overall performance; misleading on imbalanced datasets       |
| Classification | Balanced Accuracy| (Sensitivity + Specificity) / 2                | 1          | More robust for imbalanced classes                            |
| Classification | Precision        | TP / (TP + FP)                                 | 1          | When false positives are costly (e.g., spam filtering)        |
| Classification | Recall (Sensitivity)| TP / (TP + FN)                               | 1          | When false negatives are costly (e.g., medical diagnosis)     |
| Classification | Specificity      | TN / (TN + FP)                                 | 1          | True negative rate; complements recall                        |
| Classification | F1 Score         | 2 * (Precision * Recall) / (Precision + Recall)| 1          | Balances precision and recall; good for imbalanced datasets  |
| Classification | ROC-AUC          | Area under ROC curve                           | 1          | Threshold-independent measure of classification performance  |
| Classification | Log Loss         | Penalizes confident wrong predictions          | 0          | For probabilistic classifiers                                 |
| Classification | Confusion Matrix | Table showing counts of TP, FP, FN, TN         | N/A        | Detailed breakdown of errors per class                        |
| Regression     | MAE (Mean Absolute Error)| Mean of absolute differences between predicted and actual values | 0          | Simple, interpretable average error                           |
| Regression     | MSE (Mean Squared Error)| Mean of squared differences between predicted and actual values | 0          | Penalizes larger errors more than MAE                         |
| Regression     | RMSE (Root Mean Squared Error)| Square root of MSE                     | 0          | Same units as target; sensitive to outliers                  |
| Regression     | R² Score         | 1 - (Sum of squared residuals / Total sum of squares)| 1          | Proportion of variance explained                              |
| Regression     | Adjusted R²      | R² adjusted for number of predictors           | 1          | More accurate for comparing models with different features   |

---

## Quick Guide

| If you care about...                | Use this metric                          |
|-----------------------------------|-----------------------------------------|
| Binary classification             | Accuracy, F1, ROC-AUC                   |
| Imbalanced classification         | Balanced Accuracy, F1, Precision, Recall|
| Minimizing false positives        | Precision                              |
| Minimizing false negatives        | Recall (Sensitivity)                   |
| Probabilistic output quality      | Log Loss, ROC-AUC                      |
| Regression average error          | MAE                                   |
| Penalizing large errors           | MSE, RMSE                            |
| Variance explained by model       | R² Score                             |
| Comparing models with different sizes | Adjusted R²                        |
