# W2D2: Feature Scaling & Selection Documentation

## Top 5 Selected Features

SelectKBest was used with `f_classif` and `k=5` to identify the five most relevant features for the categorical target `Gender_Label`.

The selected features were:

1. **Math_Score** – Represents mathematics performance and had the highest F-score, making it the strongest selected feature for distinguishing the target classes.

2. **English_Score** – Represents English performance and showed a strong relationship with the target classes.

3. **Science_Score** – Represents science performance and also showed a significant relationship with the target.

4. **City_Delhi** – Indicates whether the student's city is Delhi and showed a measurable relationship with the target classes.

5. **City_Mumbai** – Indicates whether the student's city is Mumbai and showed a measurable relationship with the target classes.

## Feature Selection Scores

| Feature       |   F-score |
| ------------- | --------: |
| Math_Score    | 25.623529 |
| English_Score | 18.847352 |
| Science_Score | 10.964912 |
| City_Delhi    |  2.666667 |
| City_Mumbai   |  2.666667 |

## Target and Leakage Prevention

`Gender_Label` was used as the independent categorical target for demonstration.

Derived features such as `Total_Score` and `Average_Score` were excluded from the feature-selection candidate list because they are calculated directly from the subject scores and could introduce feature leakage.

## Encoding Trade-offs

- **LabelEncoder:** Converts categories into numerical labels. It is simple but can introduce an artificial order, so it is mainly suitable for target labels.
- **OneHotEncoder:** Creates separate binary columns for each category. It is suitable for nominal categories such as City because there is no natural order.
- **OrdinalEncoder:** Converts categories into ordered numerical values. It should only be used when the categories have a meaningful order.

## Scaling Trade-offs

- **StandardScaler:** Centers data around zero and scales it based on standard deviation. It can be affected by outliers.
- **MinMaxScaler:** Scales values to a fixed range, usually 0 to 1. It is sensitive to outliers.
- **RobustScaler:** Uses the median and interquartile range, making it more resistant to outliers.

## Output Evidence

- `scaling_standard_comparison.png` – Before/after distribution comparison using StandardScaler.
- SelectKBest output and feature scores were verified in the terminal.
