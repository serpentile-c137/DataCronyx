import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import numpy as np

# Load the data
data = pd.read_csv('/var/folders/hn/z7dqkrys0jb521fxp_4sv30m0000gn/T/tmp3k4x9s2m.csv')

# Separate features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LogisticRegression': LogisticRegression(random_state=42, solver='liblinear')
}

# Define hyperparameter grids
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.01, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    'LogisticRegression': {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2']
    }
}

# Model training and evaluation
best_model = None
best_accuracy = 0
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    start_time = time.time()

    # Hyperparameter tuning using RandomizedSearchCV
    param_grid = param_grids[name]
    random_search = RandomizedSearchCV(model, param_grid, cv=3, scoring='accuracy', n_iter=10, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    
    # Get the best model from RandomizedSearchCV
    best_model_rs = random_search.best_estimator_

    # Cross-validation
    cv_scores = cross_val_score(best_model_rs, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f}")

    # Prediction and evaluation on the test set
    y_pred = best_model_rs.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'cv_accuracy': cv_scores.mean(),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'model': best_model_rs
    }

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    # Select the best model based on test accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = best_model_rs

# Print results
print("\nResults:")
for name, result in results.items():
    print(f"Model: {name}")
    print(f"Cross-validation Accuracy: {result['cv_accuracy']:.4f}")
    print(f"Test Accuracy: {result['accuracy']:.4f}")
    print("Classification Report:\n", result['classification_report'])
    print("Confusion Matrix:\n", result['confusion_matrix'])
    print("-" * 50)

# Save the best model
if best_model is not None:
    with open('model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print("Best model saved as 'model.pkl'")
else:
    print("No model was trained.")