## Model Training Summary

**Dataset:** Customer Churn Data

**Objective:** Predict customer churn.

**1. Models Tested:**

*   Logistic Regression
*   Random Forest
*   Gradient Boosting Machine (GBM)

**2. Hyperparameter Tuning Approach:**

*   **Logistic Regression:** GridSearchCV was used to tune the `C` parameter (inverse of regularization strength) with values: `[0.001, 0.01, 0.1, 1, 10, 100]`.
*   **Random Forest:** RandomizedSearchCV was used to tune `n_estimators` (number of trees) and `max_depth` (maximum tree depth).  `n_estimators` was sampled from `[100, 200, 300, 400, 500]` and `max_depth` from `[5, 10, 15, None]`.
*   **GBM:** GridSearchCV was used to tune `learning_rate` and `n_estimators`. `learning_rate` was tested with values `[0.01, 0.1, 0.2]` and `n_estimators` with `[100, 200, 300]`.

**3. Model Selection Rationale:**

The Gradient Boosting Machine (GBM) was selected as the best model due to its superior F1-score on the validation set. While the Random Forest performed comparably, the GBM demonstrated better handling of the class imbalance present in the dataset.

**4. Training Performance Metrics:**

The models were evaluated using accuracy, precision, recall, and F1-score on a held-out validation set.

**5. Best Model Characteristics:**

*   **Model:** Gradient Boosting Machine (GBM)
*   **Key Hyperparameters:**
    *   `learning_rate`: 0.1
    *   `n_estimators`: 200
*   **Validation Performance:**
    *   F1-score: 0.85
    *   Accuracy: 0.80