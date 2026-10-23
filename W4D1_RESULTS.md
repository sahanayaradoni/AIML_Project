# W4D1: Model Evaluation Metrics — Results

## Model Used

Logistic Regression with StandardScaler using a Scikit-Learn Pipeline.

## Dataset

Breast Cancer Wisconsin dataset.

- Samples: 569
- Features: 30
- Classes: 2
- Training samples: 455
- Testing samples: 114

## Test Set Metrics

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.9825 |
| Precision | 0.9861 |
| Recall    | 0.9861 |
| F1-Score  | 0.9861 |
| ROC-AUC   | 0.9954 |

## Cross-Validation

### K-Fold Cross-Validation

- Folds: 5
- Mean accuracy: 0.9771
- Standard deviation: 0.0090

### Stratified K-Fold Cross-Validation

- Folds: 5
- Mean accuracy: 0.9737
- Standard deviation: 0.0166

## Learning Curve Analysis

The learning curve shows that training and validation accuracy become close as the number of training samples increases.

Final training accuracy: 0.9895

Final validation accuracy: 0.9807

Training-validation gap: 0.0088

### Diagnosis

The model appears reasonably well-fitted because the final training and validation scores are both high and the gap between them is small.

## Output Evidence

The following output files were generated:

- `w4d1_outputs/confusion_matrix.png`
- `w4d1_outputs/learning_curve.png`

## MLflow

The experiment was tracked using MLflow under:

`Cynaris_W4D1_Model_Evaluation`

The following metrics were logged:

- Test accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- K-Fold mean accuracy
- K-Fold standard deviation
- Stratified K-Fold mean accuracy
- Stratified K-Fold standard deviation

## Testing

Pytest result:

`10 passed in 10.67s`
