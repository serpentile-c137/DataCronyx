import streamlit as st

def show_home_page():
    st.title("Welcome to AutoEDA 🎉")
    st.subheader("Key Features")
    st.write("📊 **Interactive Exploration:** Explore your datasets with interactive visualizations.")
    st.write("📈 **Stunning Charts:** Visualize data with beautiful and informative charts.")
    st.write("🛠️ **Effortless Preprocessing:** Streamline data preprocessing and preparation.")

    st.subheader("Get Started with AutoEDA")
    st.write("""
        AutoEDA is your gateway to data analysis and preprocessing.
        We've simplified the process to help you make the most of your data.
    """)

    st.subheader("Who is this for?")
    st.write("""
        - Data Scientists and Analysts wanting quick data insights.
        - Professionals seeking automated exploratory data analysis.
        - Beginners eager to learn data preprocessing and visualization.
    """)

    # Include any custom CSS or styling if needed
    # st.markdown(custom_css, unsafe_allow_html=True)
