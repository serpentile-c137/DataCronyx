import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy import stats

def run_data_preprocessing(df):
    st.subheader("Data Preprocessing")

    if st.checkbox("Show dataset before preprocessing"):
        st.write(df.head())

    cols = df.columns.tolist()

    cols_to_remove = st.multiselect("Select columns to remove", cols)
    if cols_to_remove:
        df = remove_selected_columns(df, cols_to_remove)
        st.success(f"Removed columns: {cols_to_remove}")

    missing_cols = st.multiselect("Select columns to handle missing data", df.columns[df.isnull().any()].tolist())
    if missing_cols:
        method = st.selectbox("Select fill method", ['mean', 'median', 'mode'])
        df = fill_missing_data(df, missing_cols, method)
        st.success(f"Filled missing values in {missing_cols} using {method}")

    encode_cols = st.multiselect("Select columns to encode", df.select_dtypes(include=['object', 'category']).columns.tolist())
    if encode_cols:
        df = label_encode(df, encode_cols)
        st.success(f"Label encoded columns: {encode_cols}")

    scaling_cols = st.multiselect("Select numerical columns to scale", df.select_dtypes(include=['number']).columns.tolist())
    if scaling_cols:
        scale_type = st.selectbox("Select scaling method", ['StandardScaler', 'MinMaxScaler'])
        if scale_type == 'StandardScaler':
            df = standard_scale(df, scaling_cols)
        else:
            df = min_max_scale(df, scaling_cols)
        st.success(f"Applied {scale_type} on columns {scaling_cols}")

    if st.checkbox("Detect outliers"):
        col_out = st.selectbox("Select column for outlier detection", df.select_dtypes(include=['number']).columns.tolist())
        method_out = st.selectbox("Select outlier detection method", ['IQR', 'Z-Score'])

        if method_out == 'IQR':
            outliers = detect_outliers_iqr(df, col_out)
        else:
            outliers = detect_outliers_zscore(df, col_out)
        st.write(f"Detected outliers: {outliers}")

        if outliers:
            action = st.radio("Select action on outliers", ['Remove outliers', 'Replace with median', 'Do nothing'])
            if action == 'Remove outliers':
                df = remove_outliers(df, col_out, outliers)
                st.success(f"Removed outliers from {col_out}")
            elif action == 'Replace with median':
                df = transform_outliers(df, col_out, outliers)
                st.success(f"Replaced outliers in {col_out}")

    if st.checkbox("Show dataset after preprocessing"):
        st.write(df.head())

    return df

def remove_selected_columns(df, columns):
    return df.drop(columns=columns)

def fill_missing_data(df, columns, method):
    for col in columns:
        if method == 'mean':
            df[col].fillna(df[col].mean(), inplace=True)
        elif method == 'median':
            df[col].fillna(df[col].median(), inplace=True)
        elif method == 'mode':
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def label_encode(df, columns):
    encoder = LabelEncoder()
    for col in columns:
        df[col] = encoder.fit_transform(df[col])
    return df

def standard_scale(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def min_max_scale(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def detect_outliers_iqr(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].tolist()

def detect_outliers_zscore(df, col):
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    return df[col][z_scores > 3].tolist()

def remove_outliers(df, col, outliers):
    return df[~df[col].isin(outliers)]

def transform_outliers(df, col, outliers):
    median_val = df.loc[~df[col].isin(outliers), col].median()
    df.loc[df[col].isin(outliers), col] = median_val
    return df
