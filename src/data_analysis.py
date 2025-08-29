import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def run_data_exploration(df):
    st.subheader("Dataset Overview")
    display_dataset_overview(df)

    st.subheader("Missing Values Analysis")
    display_missing_values(df)

    st.subheader("Basic Statistics & Visualization")
    display_statistics_visualization(df)

    st.subheader("Distribution & Relationship Analysis")
    display_distribution_and_relationships(df)

def display_dataset_overview(df):
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")
    st.write(f"Duplicates: {df.shape[0] - df.drop_duplicates().shape[0]}")
    st.dataframe(df.head())

def display_missing_values(df):
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_data = pd.DataFrame({"Missing Count": missing_count, "Missing Percentage": missing_percentage})
    missing_data = missing_data[missing_data["Missing Count"] > 0]
    if not missing_data.empty:
        st.dataframe(missing_data.sort_values("Missing Count", ascending=False))
    else:
        st.info("No missing values detected.")

def display_statistics_visualization(df):
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.write("Numerical Columns Statistics:")
    if num_cols:
        st.dataframe(df[num_cols].describe())
    else:
        st.info("No numerical columns available.")

    st.write("Categorical Columns Value Counts:")
    if cat_cols:
        selected = st.multiselect("Choose categorical columns to visualize", cat_cols)
        for col in selected:
            vc = df[col].value_counts()
            st.bar_chart(vc)
            st.dataframe(vc)
    else:
        st.info("No categorical columns available.")

def display_distribution_and_relationships(df):
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not num_cols:
        st.info("No numerical columns for distribution analysis.")
        return

    feature = st.selectbox("Select numerical feature for distribution plot", num_cols)
    plot_type = st.selectbox("Select plot type", ['Histogram', 'Box Plot', 'Scatter Plot', 'Density Plot'])

    if plot_type == 'Histogram':
        fig = px.histogram(df, x=feature, title=f"Histogram of {feature}")
    elif plot_type == 'Box Plot':
        fig = px.box(df, y=feature, title=f"Box plot of {feature}")
    elif plot_type == 'Scatter Plot':
        fig = px.scatter(df, x=feature, y=feature, title=f"Scatter Plot of {feature}")
    elif plot_type == 'Density Plot':
        fig = px.density_contour(df, x=feature, title=f"Density plot of {feature}")
    
    st.plotly_chart(fig, use_container_width=True)

    if cat_cols:
        cat_feature = st.selectbox("Select categorical feature for analysis", cat_cols)
        chart_type = st.selectbox("Select categorical chart", ["Bar Chart", "Pie Chart", "Frequency Count"])
        if chart_type == "Bar Chart":
            fig = px.bar(df, x=cat_feature, title=f"Bar Chart of {cat_feature}")
            st.plotly_chart(fig)
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=cat_feature, title=f"Pie Chart of {cat_feature}")
            st.plotly_chart(fig)
        elif chart_type == "Frequency Count":
            counts = df[cat_feature].value_counts()
            st.dataframe(counts)
    else:
        st.info("No categorical columns for analysis.")
