import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor

# Load the dataset
try:
    df = pd.read_csv("example_dataset/titanic.csv")
except FileNotFoundError:
    print("Error: titanic.csv not found in example_dataset/. Please ensure the file exists.")
    exit()

# Preprocessing (handling missing values and categorical features)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Define features and target
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load the model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: model.pkl not found. Please ensure the file exists.")
    exit()

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
if hasattr(model, "predict_proba"): #Classification
    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC curve and AUC
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # Feature importance
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        importance_df = importance_df.sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.show()

    # SHAP values
    try:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(pd.DataFrame(X_train, columns = X.columns), 10))
        shap_values = explainer.shap_values(pd.DataFrame(X_test, columns = X.columns).iloc[0:10,:])

        shap.summary_plot(shap_values, features=pd.DataFrame(X_test, columns = X.columns).iloc[0:10,:], class_names = ['Not Survived','Survived'])
    except Exception as e:
        print(f"SHAP value calculation failed: {e}")
else: #Regression
    # Regression metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Feature importance
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        importance_df = importance_df.sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.show()

    # SHAP values
    try:
        explainer = shap.KernelExplainer(model.predict, shap.sample(pd.DataFrame(X_train, columns = X.columns), 10))
        shap_values = explainer.shap_values(pd.DataFrame(X_test, columns = X.columns).iloc[0:10,:])

        shap.summary_plot(shap_values, features=pd.DataFrame(X_test, columns = X.columns).iloc[0:10,:])
    except Exception as e:
        print(f"SHAP value calculation failed: {e}")