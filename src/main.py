import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu

import data_analysis as analysis
import data_preprocessing as preprocessing
import home

st.set_page_config(page_icon="✨", page_title="AutoEDA")

HIDE_ST_STYLE = """
<style>
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(HIDE_ST_STYLE, unsafe_allow_html=True)

def main():
    st.sidebar.title("AutoEDA: Automated Exploratory Data Analysis and Processing")

    # Option menu for navigation
    choice = option_menu(
        menu_title=None,
        options=["Home", "Data Upload", "Data Exploration", "Data Preprocessing"],
        icons=["house", "upload", "bar-chart", "gear"],
        menu_icon="cast",
        orientation="horizontal",
        default_index=0)

    if choice == "Home":
        home.show_home_page()

    elif choice == "Data Upload":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("Data Loaded Successfully!")
            st.session_state["dataset"] = df
            st.write(df.head())

    elif choice == "Data Exploration":
        if "dataset" not in st.session_state:
            st.warning("Please upload a dataset first.")
        else:
            df = st.session_state["dataset"]
            analysis.run_data_exploration(df)

    elif choice == "Data Preprocessing":
        if "dataset" not in st.session_state:
            st.warning("Please upload a dataset first.")
        else:
            df = st.session_state["dataset"]
            processed_df = preprocessing.run_data_preprocessing(df)
            st.session_state["dataset"] = processed_df  # Save processed data

if __name__ == "__main__":
    main()
