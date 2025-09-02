import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from typing import List, Tuple, Optional

# Ensure logs directory exists (date-wise)
log_date = datetime.now().strftime("%Y-%m-%d")
log_dir = os.path.join(os.path.dirname(__file__), '../logs', log_date)
os.makedirs(log_dir, exist_ok=True)

# Configure logging: daily rotating, shared logfile in date-wise folder
logfile_path = os.path.join(log_dir, 'datacronyx.log')
if not any(isinstance(h, TimedRotatingFileHandler) and getattr(h, 'baseFilename', None) == logfile_path for h in logging.getLogger().handlers):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            TimedRotatingFileHandler(logfile_path, when="midnight", backupCount=30, encoding='utf-8')
        ]
    )

def apply_pca(
    df: pd.DataFrame, 
    n_components: int
) -> Tuple[pd.DataFrame, Optional[PCA]]:
    """Apply PCA to the numerical columns of the DataFrame."""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(numeric_df)
        pc_columns = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(principal_components, columns=pc_columns, index=df.index)
        logging.info(f"PCA applied with {n_components} components.")
        return pca_df, pca
    except Exception as e:
        logging.error(f"Error applying PCA: {e}")
        st.error(f"Error applying PCA: {e}")
        return df, None

def select_k_best_features(
    df: pd.DataFrame, 
    target: pd.Series, 
    k: int, 
    problem_type: str = "classification"
) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """Select K best features using ANOVA F-value for classification or regression."""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if problem_type == "classification":
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(numeric_df, target)
        mask = selector.get_support()
        selected_features = numeric_df.columns[mask].tolist()
        kbest_df = numeric_df[selected_features]
        logging.info(f"Selected top {k} features: {selected_features}")
        return kbest_df, selected_features
    except Exception as e:
        logging.error(f"Error selecting K best features: {e}")
        st.error(f"Error selecting K best features: {e}")
        return df, None
