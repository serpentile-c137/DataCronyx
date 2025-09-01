import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import logging
from typing import Tuple, Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def split_data(
    df: pd.DataFrame, 
    target_column: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets."""
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logging.info(f"Data split: {X_train.shape}, {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        st.error(f"Error splitting data: {e}")
        return None, None, None, None

def train_classification_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    model_type: str = "Logistic Regression"
) -> Any:
    """Train a classification model."""
    try:
        if model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "Random Forest":
            model = RandomForestClassifier()
        elif model_type == "SVM":
            model = SVC(probability=True)
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier()
        else:
            st.warning("Unsupported model type. Using Logistic Regression.")
            model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        logging.info(f"Trained {model_type} classifier.")
        return model
    except Exception as e:
        logging.error(f"Error training classification model: {e}")
        st.error(f"Error training classification model: {e}")
        return None

def evaluate_classification_model(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Dict[str, Any]:
    """Evaluate a classification model and return metrics."""
    try:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logging.info(f"Classification accuracy: {acc}")
        return {"accuracy": acc, "report": report, "confusion_matrix": cm}
    except Exception as e:
        logging.error(f"Error evaluating classification model: {e}")
        st.error(f"Error evaluating classification model: {e}")
        return {}

def train_regression_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    model_type: str = "Linear Regression"
) -> Any:
    """Train a regression model."""
    try:
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Random Forest":
            model = RandomForestRegressor()
        elif model_type == "Ridge":
            model = Ridge()
        elif model_type == "Lasso":
            model = Lasso()
        elif model_type == "SVM":
            model = SVR()
        elif model_type == "Decision Tree":
            model = DecisionTreeRegressor()
        elif model_type == "Gradient Boosting":
            model = GradientBoostingRegressor()
        else:
            st.warning("Unsupported model type. Using Linear Regression.")
            model = LinearRegression()
        model.fit(X_train, y_train)
        logging.info(f"Trained {model_type} regressor.")
        return model
    except Exception as e:
        logging.error(f"Error training regression model: {e}")
        st.error(f"Error training regression model: {e}")
        return None

def evaluate_regression_model(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Dict[str, Any]:
    """Evaluate a regression model and return metrics."""
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Regression MSE: {mse}, R2: {r2}")
        return {"mse": mse, "r2": r2}
    except Exception as e:
        logging.error(f"Error evaluating regression model: {e}")
        st.error(f"Error evaluating regression model: {e}")
        return {}
