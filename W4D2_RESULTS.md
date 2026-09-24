# W4D2: Bias-Variance Tradeoff & Regularisation

## Objective

Implemented the bias-variance tradeoff and regularisation workflow using systematic hyperparameter search strategies.

## Tasks Completed

- Demonstrated the bias-variance tradeoff using polynomial model complexity.
- Compared training and testing performance to identify overfitting.
- Implemented Ridge Regression with L2 regularisation.
- Implemented Lasso Regression with L1 regularisation.
- Used `GridSearchCV` for systematic hyperparameter tuning.
- Used `RandomizedSearchCV` for efficient hyperparameter tuning.
- Compared the selected hyperparameters and model performance.
- Added MLflow experiment tracking for parameters and evaluation metrics.
- Created automated tests using pytest.

## Bias-Variance Results

| Polynomial Degree | Train RMSE | Test RMSE | Train R² | Test R² |
| ----------------: | ---------: | --------: | -------: | ------: |
|                 1 |    19.0710 |   21.0989 |   0.9718 |  0.9635 |
|                 2 |    18.0658 |   23.7875 |   0.9747 |  0.9536 |
|                 5 |     2.7135 |   79.8799 |   0.9994 |  0.4763 |
|                10 |     2.3196 |  260.8507 |   0.9996 | -4.5846 |

The higher-degree models achieve extremely low training error but substantially higher testing error. This demonstrates high variance and overfitting.

## Regularisation Results

| Model                    | Best Alpha | Test RMSE | Test R² |
| ------------------------ | ---------: | --------: | ------: |
| Ridge GridSearchCV       |       0.01 |   21.0775 |  0.9635 |
| Lasso GridSearchCV       |        0.1 |   21.1126 |  0.9634 |
| Ridge RandomizedSearchCV |      0.001 |   21.0774 |  0.9635 |

Regularisation improves generalisation by penalising excessive model complexity.

## Grid Search vs Randomized Search

### GridSearchCV

Grid Search evaluates every specified hyperparameter combination. It is suitable when the search space is small and the candidate values are carefully defined.

### RandomizedSearchCV

Randomized Search samples a specified number of combinations from the search space. It is useful when the search space is large because it can explore a wider range with fewer evaluations.

## Hyperparameter Interpretation

### Ridge `alpha`

Controls the strength of L2 regularisation.

- Smaller `alpha` → weaker regularisation.
- Larger `alpha` → stronger regularisation.

### Lasso `alpha`

Controls the strength of L1 regularisation.

- Smaller `alpha` → weaker regularisation.
- Larger `alpha` → stronger regularisation.
- L1 regularisation can drive some coefficients toward zero.

## MLflow

An MLflow experiment named:

`W4D2_Bias_Variance_Regularisation`

was created.

The workflow logged:

- Random state
- Cross-validation folds
- Ridge test RMSE and R²
- Lasso test RMSE and R²
- Randomized Search test RMSE and R²
- Best Ridge alpha
- Best Lasso alpha
- Best Randomized Search alpha
- Bias-variance results artifact
- Regularisation comparison artifact

## Testing

Pytest execution:

`7 passed in 23.64s`

All seven W4D2 tests passed successfully.

## Output Files

The following output files were generated:

```text
w4d2_outputs/
├── bias_variance_results.csv
└── regularisation_comparison.csv
```

## Key Learning

The experiment shows that a model can perform extremely well on training data while performing poorly on unseen data. This is a high-variance/overfitting situation. Regularisation and systematic hyperparameter tuning help control model complexity and improve generalisation.

## Viva Preparation

### 1. Explain what you built today and why you made your key design decisions.

I built a regression workflow demonstrating the bias-variance tradeoff and regularisation. I compared different polynomial complexities to show underfitting and overfitting, then implemented Ridge and Lasso regularisation. I used GridSearchCV and RandomizedSearchCV instead of manually guessing hyperparameters because systematic search provides a reproducible way to select suitable values. MLflow was used to track important parameters and evaluation metrics.

### 2. What was the hardest part? How did you solve it?

The hardest part was understanding how model complexity affects the difference between training and testing performance. I solved this by comparing multiple polynomial degrees and evaluating both training and testing RMSE and R². I then used regularisation and cross-validated hyperparameter search to control model complexity.

### 3. If you had one more day, what would you improve?

I would expand the hyperparameter search space, add more visualisations of the bias-variance tradeoff, compare additional regularised models, and improve MLflow experiment comparison and reporting.
