import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('../example_dataset/titanic.csv')

# Handle missing values (simplified)
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Convert categorical features to numerical (one-hot encoding)
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Select features and target variable
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
target = 'Survived'

X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LogisticRegression': LogisticRegression(random_state=42)
}

# Define hyperparameter grids
param_grids = {
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [4, 6, 8],
        'min_samples_split': [2, 4]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    },
    'LogisticRegression': {
        'C': [0.1, 1.0, 10.0],
        'solver': ['liblinear']
    }
}

# Perform hyperparameter tuning and cross-validation
best_models = {}
training_metrics = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f'Tuning {name}...')
    grid_search = GridSearchCV(model, param_grids[name], cv=skf, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f'Best parameters for {name}: {grid_search.best_params_}')

    # Evaluate on test set
    y_pred = best_models[name].predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    training_metrics[name] = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': confusion
    }

# Model comparison
best_model_name = max(training_metrics, key=lambda k: training_metrics[k]['accuracy'])
best_model = best_models[best_model_name]

print(f'Best model: {best_model_name} with accuracy {training_metrics[best_model_name]["accuracy"]}')

# Save the best model
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Print training metrics
print('\nTraining Metrics:')
for name, metrics in training_metrics.items():
    print(f'\nModel: {name}')
    print(f'Accuracy: {metrics["accuracy"]}')
    print(f'Classification Report:\n{metrics["classification_report"]}')
    print(f'Confusion Matrix:\n{metrics["confusion_matrix"]}')