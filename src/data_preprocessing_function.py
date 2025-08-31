import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import logging
from typing import List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def remove_selected_columns(df: pd.DataFrame, columns_remove: List[str]) -> pd.DataFrame:
    """Remove selected columns from the DataFrame."""
    try:
        result = df.drop(columns=columns_remove)
        logging.info(f"Removed columns: {columns_remove}")
        return result
    except Exception as e:
        logging.error(f"Error removing columns {columns_remove}: {e}")
        st.error(f"Error removing columns: {e}")
        return df

def remove_rows_with_missing_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Remove rows with missing values in specific columns."""
    try:
        if columns:
            result = df.dropna(subset=columns)
            logging.info(f"Removed rows with missing data in columns: {columns}")
            return result
        return df
    except Exception as e:
        logging.error(f"Error removing rows with missing data in {columns}: {e}")
        st.error(f"Error removing rows with missing data: {e}")
        return df

def fill_missing_data(df: pd.DataFrame, columns: List[str], method: str) -> pd.DataFrame:
    """Fill missing data in specified columns using mean, median, or mode."""
    try:
        for column in columns:
            if method == 'mean':
                df[column].fillna(df[column].mean(), inplace=True)
            elif method == 'median':
                df[column].fillna(df[column].median(), inplace=True)
            elif method == 'mode':
                mode_val = df[column].mode().iloc[0]
                df[column].fillna(mode_val, inplace=True)
            else:
                logging.warning(f"Unknown fill method: {method}")
        logging.info(f"Filled missing data in columns {columns} using {method}")
        return df
    except Exception as e:
        logging.error(f"Error filling missing data in {columns} with {method}: {e}")
        st.error(f"Error filling missing data: {e}")
        return df

def one_hot_encode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Apply one-hot encoding to specified columns."""
    try:
        result = pd.get_dummies(df, columns=columns, prefix=columns, drop_first=False)
        logging.info(f"One-hot encoded columns: {columns}")
        return result
    except Exception as e:
        logging.error(f"Error one-hot encoding columns {columns}: {e}")
        st.error(f"Error one-hot encoding columns: {e}")
        return df

def label_encode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Apply label encoding to specified columns."""
    try:
        label_encoder = LabelEncoder()
        for col in columns:
            df[col] = label_encoder.fit_transform(df[col])
        logging.info(f"Label encoded columns: {columns}")
        return df
    except Exception as e:
        logging.error(f"Error label encoding columns {columns}: {e}")
        st.error(f"Error label encoding columns: {e}")
        return df

def standard_scale(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Standard scale specified columns."""
    try:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        logging.info(f"Standard scaled columns: {columns}")
        return df
    except Exception as e:
        logging.error(f"Error standard scaling columns {columns}: {e}")
        st.error(f"Error standard scaling columns: {e}")
        return df

def min_max_scale(df: pd.DataFrame, columns: List[str], feature_range: tuple = (0, 1)) -> pd.DataFrame:
    """Min-max scale specified columns."""
    try:
        scaler = MinMaxScaler(feature_range=feature_range)
        df[columns] = scaler.fit_transform(df[columns])
        logging.info(f"Min-max scaled columns: {columns} with range {feature_range}")
        return df
    except Exception as e:
        logging.error(f"Error min-max scaling columns {columns}: {e}")
        st.error(f"Error min-max scaling columns: {e}")
        return df

def detect_outliers_iqr(df: pd.DataFrame, column_name: str) -> List[Any]:
    """Detect outliers in a column using the IQR method."""
    try:
        data = df[column_name]
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        outliers.sort()
        logging.info(f"Detected {len(outliers)} outliers in {column_name} using IQR")
        return outliers
    except Exception as e:
        logging.error(f"Error detecting outliers in {column_name} using IQR: {e}")
        st.error(f"Error detecting outliers using IQR: {e}")
        return []

def detect_outliers_zscore(df: pd.DataFrame, column_name: str) -> List[Any]:
    """Detect outliers in a column using the z-score method."""
    try:
        data = df[column_name]
        z_scores = np.abs(stats.zscore(data))
        threshold = 3
        outliers = [data.iloc[i] for i in range(len(data)) if z_scores[i] > threshold]
        logging.info(f"Detected {len(outliers)} outliers in {column_name} using z-score")
        return outliers
    except Exception as e:
        logging.error(f"Error detecting outliers in {column_name} using z-score: {e}")
        st.error(f"Error detecting outliers using z-score: {e}")
        return []

def remove_outliers(df: pd.DataFrame, column_name: str, outliers: List[Any]) -> pd.DataFrame:
    """Remove rows where column value is in outliers."""
    try:
        result = df[~df[column_name].isin(outliers)]
        logging.info(f"Removed {len(outliers)} outliers from {column_name}")
        return result
    except Exception as e:
        logging.error(f"Error removing outliers from {column_name}: {e}")
        st.error(f"Error removing outliers: {e}")
        return df

def transform_outliers(df: pd.DataFrame, column_name: str, outliers: List[Any]) -> pd.DataFrame:
    """Replace outlier values with the median of non-outlier values."""
    try:
        non_outliers = df[~df[column_name].isin(outliers)]
        median_value = non_outliers[column_name].median()
        df.loc[df[column_name].isin(outliers), column_name] = median_value
        logging.info(f"Transformed {len(outliers)} outliers in {column_name} to median value {median_value}")
        return df
    except Exception as e:
        logging.error(f"Error transforming outliers in {column_name}: {e}")
        st.error(f"Error transforming outliers: {e}")
        return df
