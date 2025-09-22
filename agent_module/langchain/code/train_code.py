import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import time
import numpy as np

# Load the dataset
data = pd.read_csv("/var/folders/hn/z7dqkrys0jb521fxp_4sv30m0000gn/T/tmpjh1kms7r.csv")

# Separate features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LogisticRegression': LogisticRegression(random_state=42, solver='liblinear')
}

# Define the hyperparameter grids
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'min_samples_split': [2, 4]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.01]
    },
    'LogisticRegression': {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2']
    }
}

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_accuracy = 0
best_model_name = None
training_metrics = {}

# Iterate through the models
for model_name, model in models.items():
    print(f"Training {model_name}...")
    start_time = time.time()

    # Hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(model, param_grids[model_name], scoring='accuracy', cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model from GridSearchCV
    best_model_cv = grid_search.best_estimator_

    # Evaluate the best model on the test set
    y_pred = best_model_cv.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{model_name} Accuracy: {accuracy}")

    # Generate classification report and confusion matrix
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    training_time = time.time() - start_time

    training_metrics[model_name] = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': matrix,
        'best_params': grid_search.best_params_,
        'training_time': training_time
    }

    # Update the best model if current model is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = best_model_cv
        best_model_name = model_name

print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy}")

# Save the best model
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("Training complete. Best model saved as 'model.pkl'")

# Print training metrics
print("\nTraining Metrics:")
for model_name, metrics in training_metrics.items():
    print(f"\nModel: {model_name}")
    print(f"  Accuracy: {metrics['accuracy']}")
    print(f"  Best Parameters: {metrics['best_params']}")
    print(f"  Training Time: {metrics['training_time']:.2f} seconds")
    print("  Classification Report:\n", metrics['classification_report'])
    print("  Confusion Matrix:\n", metrics['confusion_matrix'])