import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
from sklearn.inspection import permutation_importance

# Load the dataset
data = pd.read_csv("/var/folders/hn/z7dqkrys0jb521fxp_4sv30m0000gn/T/tmpjh1kms7r.csv")

# Preprocessing (handle missing values, encoding, scaling) - Adapt based on your data
for col in data.columns:
    if data[col].dtype == 'object':
        try:
            data[col] = pd.to_numeric(data[col])
        except ValueError:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

data = data.fillna(data.mean())

# Identify target and features - Adapt based on your data
target_column = data.columns[-1]
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make predictions
y_pred = model.predict(X_test)

# Model Evaluation
if 'predict_proba' in dir(model):  # Classification
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))

    except Exception as e:
        print(f"Error during classification evaluation: {e}")
        print("Accuracy:", accuracy_score(y_test, y_pred))

else:  # Regression
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values")
    plt.show()

# Feature Importance (Permutation Importance)
try:
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    importance = result.importances_mean

    plt.figure(figsize=(10, 6))
    plt.bar(X.columns, importance)
    plt.xticks(rotation=90)
    plt.title('Permutation Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error during permutation importance: {e}")

# SHAP values
try:
    explainer = shap.KernelExplainer(model.predict, X_train[:50])
    shap_values = explainer.shap_values(X_test[:50])
    shap.summary_plot(shap_values, X_test[:50], feature_names=X.columns)

except Exception as e:
    print(f"Error during SHAP analysis: {e}")