import streamlit as st
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

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

def show_home_page() -> None:
    """Display the home page content for DataCronyx."""
    try:
        # Inject Dark Theme CSS
        st.markdown(custom_css(), unsafe_allow_html=True)

        # Key Features
        st.subheader("Key Features")
        st.write("📊 **Interactive Exploration:** Explore your datasets with interactive visualizations.")
        st.write("📈 **Stunning Charts:** Visualize data with beautiful and informative charts.")
        st.write("🛠️ **Effortless Preprocessing:** Streamline data preprocessing and preparation.")

        # Get Started Section
        st.subheader("Get Started with DataCronyx")
        st.write("DataCronyx is your gateway to data analysis and preprocessing. We've simplified the process to help you make the most of your data.")

        # Target Audience
        st.markdown('''
        <div class="target-audience">
            <div class="audience">
                <div class="audience-icon">📊</div>
                <div class="audience-title">Data Analysts</div>
            </div>
            <div class="audience">
                <div class="audience-icon">🔎</div>
                <div class="audience-title">Data Scientists</div>
            </div>
            <div class="audience">
                <div class="audience-icon">🧐</div>
                <div class="audience-title">Business Professionals</div>
            </div>
            <div class="audience">
                <div class="audience-icon">📈</div>
                <div class="audience-title">Students and Educators</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Example Dataset
        st.subheader("Try it Out!")
        st.write("Get started by uploading your own dataset or use the example dataset included in sidebar. Select it and let DataCronyx do the rest!")

        logging.info("Home page displayed successfully.")
    except Exception as e:
        logging.error(f"Error displaying home page: {e}")
        st.error(f"Error displaying home page: {e}")

def custom_css() -> str:
    """Return custom dark theme CSS styles as a string."""
    try:
        custom_css = """
        <style>
        body {
            background-color: #0e0e0e;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            color: #f5f5f5;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            text-align: center;
            padding: 40px;
        }

        .header {
            font-size: 48px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 16px;
        }

        .tagline {
            font-size: 24px;
            color: #bbbbbb;
            margin-bottom: 32px;
        }

        .features {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            margin-bottom: 40px;
        }

        .feature {
            flex: 1;
            text-align: center;
            padding: 20px;
            background-color: #1a1a1a;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.6);
            margin: 8px;
            transition: transform 0.3s ease-in-out, background-color 0.3s;
            color: #f5f5f5;
        }

        .feature:hover {
            transform: scale(1.05);
            background-color: #262626;
        }

        .feature-icon {
            font-size: 36px;
            color: #4CAF50;
        }

        .feature-title {
            font-size: 18px;
            font-weight: bold;
            margin-top: 16px;
            color: #ffffff;
        }

        .action-button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 16px 32px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        .action-button:hover {
            background-color: #45a049;
        }

        /* Target Audience Cards */
        .target-audience {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        .audience {
            flex: 0 1 calc(50% - 10px);
            background-color: #1a1a1a;
            border-radius: 10px;
            margin: 5px;
            padding: 20px;
            text-align: center;
            color: #f5f5f5;
            box-shadow: 0 4px 12px rgba(0,0,0,0.6);
            transition: background-color 0.3s;
        }
        .audience:hover {
            background-color: #262626;
        }
        .audience-icon {
            font-size: 2em;
        }
        .audience-title {
            margin-top: 10px;
            font-size: 16px;
            font-weight: bold;
        }

        /* Override Streamlit default text colors */
        h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {
            color: #f5f5f5 !important;
        }
        </style>
        """
        logging.info("Custom dark theme CSS generated successfully.")
        return custom_css
    except Exception as e:
        logging.error(f"Error generating custom CSS: {e}")
        return ""

# Run only if this script is executed directly
if __name__ == "__main__":
    show_home_page()
