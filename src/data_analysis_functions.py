''' This file contains all the functions that are used in the main file. 
This is so as to reduce the clutter in the main file and isolate the core functionalites of the application in seprate file
'''

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px
import logging
from typing import List, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Function to load the csv data to a dataframe
def load_data(file: Any) -> pd.DataFrame:
    """Load CSV data from a file-like object or path into a DataFrame."""
    try:
        df = pd.read_csv(file)
        logging.info(f"Loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Function to find categorical and numerical columns/variables in dataset
def categorical_numerical(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify numerical and categorical columns in the DataFrame."""
    num_columns, cat_columns = [], []
    for col in df.columns:
        try:
            if len(df[col].unique()) <= 30 or df[col].dtype == np.object_:
                cat_columns.append(col.strip())
            else:
                num_columns.append(col.strip())
        except Exception as e:
            logging.warning(f"Error processing column {col}: {e}")
    logging.info(f"Identified {len(num_columns)} numerical and {len(cat_columns)} categorical columns")
    return num_columns, cat_columns

# Function to display dataset overview
def display_dataset_overview(df: pd.DataFrame, cat_columns: List[str], num_columns: List[str]) -> None:
    """Display an overview of the dataset using Streamlit."""
    try:
        display_rows = st.slider("Display Rows", 1, len(df), len(df) if len(df) < 20 else 20)
        st.write(df.head(display_rows))
        st.subheader("2. Dataset Overview")
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write(f"**Duplicates:** {df.shape[0] - df.drop_duplicates().shape[0]}")
        st.write(f"**Categorical Columns:** {len(cat_columns)}")
        st.write(cat_columns)
        st.write(f"**Numerical Columns:** {len(num_columns)}")
        st.write(num_columns)
    except Exception as e:
        logging.error(f"Error displaying dataset overview: {e}")
        st.error(f"Error displaying dataset overview: {e}")

# Function to find the missing values in the dataset
def display_missing_values(df: pd.DataFrame) -> None:
    """Display missing value counts and percentages for each column."""
    try:
        missing_count = df.isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        missing_data = pd.DataFrame({'Missing Count': missing_count, 'Missing Percentage': missing_percentage})
        missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
        if not missing_data.empty:
            st.write("Missing Data Summary:")
            st.write(missing_data)
        else:
            st.info("No Missing Value present in the Dataset")
    except Exception as e:
        logging.error(f"Error displaying missing values: {e}")
        st.error(f"Error displaying missing values: {e}")

# Function to display basic statistics and visualizations about the dataset
def display_statistics_visualization(df: pd.DataFrame, cat_columns: List[str], num_columns: List[str]) -> None:
    """Display summary statistics and visualizations for numerical and categorical columns."""
    try:
        st.write("Summary Statistics for Numerical Columns")
        if len(num_columns) != 0:
            num_df = df[num_columns]
            st.write(num_df.describe())
        else:
            st.info("The dataset does not have any numerical columns")

        st.write("Statistics for Categorical Columns")
        if len(cat_columns) != 0:
            num_cat_columns = st.number_input("Select the number of categorical columns to visualize:", min_value=1, max_value=len(cat_columns))
            selected_cat_columns = st.multiselect("Select the Categorical Columns for bar chart", cat_columns, cat_columns[:num_cat_columns])
            for column in selected_cat_columns:
                st.write(f"**{column}**")
                value_counts = df[column].value_counts()
                st.bar_chart(value_counts)
                st.write(f"Value Count for {column}")
                value_counts_table = df[column].value_counts().reset_index()
                value_counts_table.columns = ['Value', 'Count']
                st.write(value_counts_table)
        else:
            st.info("The dataset does not have any categorical columns")
    except Exception as e:
        logging.error(f"Error displaying statistics/visualization: {e}")
        st.error(f"Error displaying statistics/visualization: {e}")

# Funciton to display the datatypes
def display_data_types(df: pd.DataFrame) -> None:
    """Display the data types of each column in the DataFrame."""
    try:
        data_types_df = pd.DataFrame({'Data Type': df.dtypes})
        st.write(data_types_df)
    except Exception as e:
        logging.error(f"Error displaying data types: {e}")
        st.error(f"Error displaying data types: {e}")

# Function to search for a particular column or particular datatype in the dataset
def search_column(df: pd.DataFrame) -> None:
    """Search for columns by name or filter by data type."""
    try:
        search_query = st.text_input("Search for a column:")
        selected_data_type = st.selectbox("Filter by Data Type:", ['All'] + [str(dt) for dt in df.dtypes.unique().tolist()])
        filtered_df = df.copy()
        if search_query:
            filtered_df = filtered_df.loc[:, filtered_df.columns.str.contains(search_query, case=False)]
        if selected_data_type != 'All':
            filtered_df = filtered_df.select_dtypes(include=[selected_data_type])
        st.write(filtered_df)
    except Exception as e:
        logging.error(f"Error searching columns: {e}")
        st.error(f"Error searching columns: {e}")

## FUNCTIONS FOR TAB2: Data Exploration and Visualization

def display_individual_feature_distribution(df: pd.DataFrame, num_columns: List[str]) -> None:
    """Display distribution plots and statistics for a selected numerical feature."""
    try:
        st.subheader("Analyze Individual Feature Distribution")
        st.markdown("Here, you can explore individual numerical features, visualize their distributions, and analyze relationships between features.")
        if len(num_columns) == 0:
            st.info("The dataset does not have any numerical columns")
            return
        st.write("#### Understanding Numerical Features")
        feature = st.selectbox(label="Select Numerical Feature", options=num_columns, index=0)
        df_description = df.describe()
        null_count = df[feature].isnull().sum()
        st.write("Count: ", df_description[feature]['count'])
        st.write("Missing Count: ", null_count)
        st.write("Mean: ", df_description[feature]['mean'])
        st.write("Standard Deviation: ", df_description[feature]['std'])
        st.write("Minimum: ", df_description[feature]['min'])
        st.write("Maximum: ", df_description[feature]['max'])
        st.subheader("Distribution Plots")
        plot_type = st.selectbox(label="Select Plot Type", options=['Histogram', 'Scatter Plot', 'Density Plot', 'Box Plot'])
        fig = None
        if plot_type == 'Histogram':
            fig = px.histogram(df, x=feature, title=f'Histogram of {feature}')
        elif plot_type == 'Scatter Plot':
            fig = px.scatter(df, x=feature, y=feature, title=f'Scatter plot of {feature}')
        elif plot_type == 'Density Plot':
            fig = px.density_contour(df, x=feature, title=f'Density plot of {feature}')
        elif plot_type == 'Box Plot':
            fig = px.box(df, y=feature, title=f'Box plot of {feature}')
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logging.error(f"Error displaying feature distribution: {e}")
        st.error(f"Error displaying feature distribution: {e}")

def display_scatter_plot_of_two_numeric_features(df: pd.DataFrame, num_columns: List[str]) -> None:
    """Display a scatter plot for two selected numerical features."""
    try:
        if len(num_columns) == 0:
            st.info("The dataset does not have any numerical columns")
            return
        if len(num_columns) != 0:
            x_feature = st.selectbox(label="Select X-Axis Feature", options=num_columns, index=0)
            y_feature = st.selectbox(label="Select Y-Axis Feature", options=num_columns, index=1 if len(num_columns) > 1 else 0)
            scatter_fig = px.scatter(df, x=x_feature, y=y_feature, title=f'Scatter Plot: {x_feature} vs {y_feature}')
            st.plotly_chart(scatter_fig, use_container_width=True)
    except Exception as e:
        logging.error(f"Error displaying scatter plot: {e}")
        st.error(f"Error displaying scatter plot: {e}")

def categorical_variable_analysis(df: pd.DataFrame, cat_columns: List[str]) -> None:
    """Analyze and visualize a selected categorical variable."""
    try:
        if not cat_columns:
            st.info("No categorical columns available.")
            return
        categorical_feature = st.selectbox(label="Select Categorical Feature", options=cat_columns)
        categorical_plot_type = st.selectbox(label="Select Plot Type", options=["Bar Chart", "Pie Chart", "Stacked Bar Chart", "Frequency Count"])
        fig = None
        if categorical_plot_type == "Bar Chart":
            fig = px.bar(df, x=categorical_feature, title=f"Bar Chart of {categorical_feature}")
        elif categorical_plot_type == "Pie Chart":
            fig = px.pie(df, names=categorical_feature, title=f"Pie Chart of {categorical_feature}")
        elif categorical_plot_type == "Stacked Bar Chart":
            st.write("Select a second categorical feature for stacking")
            second_categorical_feature = st.selectbox(label="Select Second Categorical Feature", options=cat_columns)
            fig = px.bar(df, x=categorical_feature, color=second_categorical_feature, title=f"Stacked Bar Chart of {categorical_feature} by {second_categorical_feature}")
        elif categorical_plot_type == "Frequency Count":
            cat_value_counts = df[categorical_feature].value_counts()
            st.write(f"Frequency Count for {categorical_feature}: ")
            st.write(cat_value_counts)
        if categorical_plot_type != "Frequency Count" and fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logging.error(f"Error in categorical variable analysis: {e}")
        st.error(f"Error in categorical variable analysis: {e}")

def feature_exploration_numerical_variables(df: pd.DataFrame, num_columns: List[str]) -> None:
    """Explore relationships between selected numerical features."""
    try:
        selected_features = st.multiselect("Select Features for Exploration:", num_columns, default=num_columns[:2], key="feature_exploration")
        if len(selected_features) < 2:
            st.warning("Please select at least two numerical features for exploration.")
        else:
            st.subheader("Explore Relationships Between Features")
            # Scatter Plot Matrix
            if st.button("Generate Scatter Plot Matrix"):
                scatter_matrix_fig = px.scatter_matrix(df, dimensions=selected_features, title="Scatter Plot Matrix")
                st.plotly_chart(scatter_matrix_fig, use_container_width=True)
            # Pair Plot
            if st.button("Generate Pair Plot"):
                pair_plot_fig = sns.pairplot(df[selected_features])
                st.pyplot(pair_plot_fig)
            # Correlation Heatmap
            if st.button("Generate Correlation Heatmap"):
                correlation_matrix = df[selected_features].corr()
                plt.figure(figsize=(10, 6))
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
                plt.title("Correlation Heatmap")
                st.pyplot(plt)
    except Exception as e:
        logging.error(f"Error in feature exploration: {e}")
        st.error(f"Error in feature exploration: {e}")

def categorical_numerical_variable_analysis(df: pd.DataFrame, cat_columns: List[str], num_columns: List[str]) -> None:
    """Analyze the relationship between a categorical and a numerical variable."""
    try:
        if not cat_columns or not num_columns:
            st.warning("Both categorical and numerical columns are required.")
            return
        categorical_feature_1 = st.selectbox(label="Categorical Feature", options=cat_columns)        
        numerical_feature_1 = st.selectbox(label="Numerical Feature", options=num_columns)
        group_data = df.groupby(categorical_feature_1)[numerical_feature_1].mean().reset_index()
        st.subheader("Relationship between Categorical and Numerical Variables")
        st.write(f"Mean {numerical_feature_1} by {categorical_feature_1}")
        fig = px.bar(group_data, x=categorical_feature_1, y=numerical_feature_1, title=f"{numerical_feature_1} by {categorical_feature_1}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logging.error(f"Error in categorical-numerical variable analysis: {e}")
        st.error(f"Error in categorical-numerical variable analysis: {e}")
