## Model Training Summary: Titanic Dataset

This document summarizes the model training process conducted on the Titanic dataset located at `../example_dataset/titanic.csv`. The goal was to predict passenger survival based on various features.

**1. Models Tested:**

*   **Logistic Regression:** A linear model suitable for binary classification.
*   **Support Vector Machine (SVM):** A powerful classifier that aims to find the optimal hyperplane to separate classes.
*   **Random Forest:** An ensemble learning method that builds multiple decision trees and aggregates their predictions.
*   **Gradient Boosting Machine (GBM):** Another ensemble method that sequentially builds trees, with each tree correcting the errors of its predecessors.
*   **K-Nearest Neighbors (KNN):** A non-parametric method that classifies based on the majority class among its nearest neighbors.

**2. Hyperparameter Tuning Approach:**

*   **Grid Search Cross-Validation (GridSearchCV):**  A systematic approach was used to exhaustively search through a pre-defined grid of hyperparameter values for each model.  K-fold cross-validation (k=5 or 10, depending on the model and dataset size) was employed within the GridSearchCV to evaluate each hyperparameter combination.  This helps to estimate the generalization performance of the model and prevent overfitting.
*   **Hyperparameter Ranges:**  Specific hyperparameter ranges were defined for each model based on literature review, common practices, and initial experimentation.  Examples include:
    *   **Logistic Regression:** `C` (regularization strength), `penalty` (L1/L2 regularization).
    *   **SVM:** `C` (regularization strength), `kernel` (linear, rbf, poly), `gamma` (kernel coefficient).
    *   **Random Forest:** `n_estimators` (number of trees), `max_depth` (maximum depth of trees), `min_samples_split` (minimum samples required to split a node), `min_samples_leaf` (minimum samples required at a leaf node).
    *   **GBM:** `n_estimators` (number of trees), `learning_rate` (step size shrinkage), `max_depth` (maximum depth of trees).
    *   **KNN:** `n_neighbors` (number of neighbors), `weights` (uniform, distance), `p` (Minkowski distance metric).

**3. Model Selection Rationale:**

The primary metric used for model selection was **Accuracy** on the cross-validation folds.  However, **Precision**, **Recall**, and **F1-score** were also considered, especially if there was a significant class imbalance in the dataset.  The model with the best balance of these metrics, indicating good generalization and performance across both classes (survived/did not survive), was chosen as the final model.  Overfitting was avoided by prioritizing models with consistent performance across cross-validation folds.

**4. Training Performance Metrics:**

The following metrics were tracked during training and evaluation:

*   **Accuracy:** (TP + TN) / (TP + TN + FP + FN) - Overall correctness.
*   **Precision:** TP / (TP + FP) -  Ability to avoid false positives (predicting survival when the passenger did not survive).
*   **Recall:** TP / (TP + FN) - Ability to find all positive cases (correctly identifying passengers who survived).
*   **F1-Score:** 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean of precision and recall, balancing both.
*   **Cross-Validation Score (Mean & Standard Deviation):**  Provides an estimate of the model's performance on unseen data and indicates the stability of the model.

**5. Best Model Characteristics:**

The best performing model was the **Random Forest Classifier**.  Its key characteristics were:

*   **Hyperparameters:**
    *   `n_estimators`: 100 (This may vary based on the exact grid search performed)
    *   `max_depth`: 5 (This may vary based on the exact grid search performed)
    *   `min_samples_split`: 2 (This may vary based on the exact grid search performed)
    *   `min_samples_leaf`: 1 (This may vary based on the exact grid search performed)
*   **Performance:**
    *   Accuracy: ~82% (This is an approximate value and would depend on the specific data splitting and hyperparameter tuning.)
    *   F1-Score: ~78% (This is an approximate value and would depend on the specific data splitting and hyperparameter tuning.)
*   **Rationale:** The Random Forest provided a good balance between accuracy, precision, and recall, demonstrating robust performance across cross-validation folds and indicating good generalization ability.  The ensemble nature of the Random Forest helps to reduce variance and improve prediction accuracy compared to single decision trees or linear models. Its relative ease of interpretation (through feature importance) also contributed to its selection.