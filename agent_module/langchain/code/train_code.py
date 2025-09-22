import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv("example_dataset/insurance.csv")

# Preprocessing & Feature Engineering
data['age_squared'] = data['age']**2
data['bmi_squared'] = data['bmi']**2
data['age_bmi_interaction'] = data['age'] * data['bmi']
data['children_squared'] = data['children']**2
data['children_age_interaction'] = data['children'] * data['age']

data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

X = data.drop('charges', axis=1)
y = data['charges']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Regression Models ---
models = {
    'RandomForestRegressor': (RandomForestRegressor(), {'n_estimators': [100, 200], 'max_depth': [5, 10]}),
    'XGBRegressor': (XGBRegressor(), {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.01]})
}

best_model = None
best_score = float('-inf')
best_model_name = None

for model_name, (model, param_grid) in models.items():
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    if grid_search.best_score_ > best_score:
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        best_model_name = model_name

print(f"Best Regression Model: {best_model_name} with R2 score: {best_score}")

# Evaluate best regression model on test set
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test R2: {r2}, RMSE: {rmse}")

# --- Classification Models (Dummy target for demonstration) ---
#Creating a dummy classification target
y_class = (data['charges'] > data['charges'].median()).astype(int)
X_class = data.drop('charges', axis=1)
X_class_scaled = scaler.fit_transform(X_class)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class_scaled, y_class, test_size=0.2, random_state=42)


classification_models = {
    'LogisticRegression': (LogisticRegression(), {'C': [0.1, 1.0], 'solver': ['liblinear']}),
    'RandomForestClassifier': (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [5, 10]}),
    'XGBClassifier': (XGBClassifier(), {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.01]})
}

best_class_model = None
best_class_score = float('-inf')
best_class_model_name = None

for model_name, (model, param_grid) in classification_models.items():
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_class, y_train_class)
    
    if grid_search.best_score_ > best_class_score:
        best_class_score = grid_search.best_score_
        best_class_model = grid_search.best_estimator_
        best_class_model_name = model_name

print(f"Best Classification Model: {best_class_model_name} with Accuracy: {best_class_score}")

# Evaluate best classification model on test set
y_pred_class = best_class_model.predict(X_test_class)
accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)
roc_auc = roc_auc_score(y_test_class, best_class_model.predict_proba(X_test_class)[:, 1])

print(f"Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, ROC AUC: {roc_auc}")


# Save the best regression model
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)