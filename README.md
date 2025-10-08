# DataCronyx

DataCronyx is an automated exploratory data analysis (EDA) and machine learning platform built with Streamlit. It provides an interactive interface for data exploration, preprocessing, feature engineering, and model training.

## Features

- **Sidebar Navigation:** All main actions (EDA, preprocessing, feature engineering, model training) are accessible from the always-open sidebar.
- **Interactive EDA:** Visualize and explore your datasets with a variety of charts and statistics.
- **Data Preprocessing:** Handle missing values, encode categorical variables, scale features, and manage outliers.
- **Feature Engineering:** Apply PCA and K-Best feature selection.
- **Model Training:** Train and evaluate multiple classification and regression models, and download trained models.
- **Sample Datasets:** Use built-in sample datasets for quick experimentation:
  - **Titanic** (Classification)
  - **Insurance** (Regression)
- **Custom Dataset Support:** Upload your own CSV or XLS files.

## Sample Datasets

- `example_dataset/titanic.csv` - For classification tasks.
- `example_dataset/insurance.csv` - For regression tasks.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/DataCronyx.git
   cd DataCronyx
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   # DataCronyx

   DataCronyx is an automated exploratory data analysis (EDA) and machine learning toolkit with an interactive Streamlit-based UI. The project bundles utilities for data exploration, preprocessing, feature engineering, model training and small automation agents used during development.

   ## Quick overview

   - Purpose: speed up EDA and baseline model experiments on tabular datasets (classification and regression).
   - UI: built with Streamlit. Main app entry: `src/main.py`.

   ## Project layout

   - `src/` - Main Streamlit app and helper modules used by the UI:
     - `data_analysis_functions.py`, `data_preprocessing_function.py`, `feature_engineering.py`, `model_training.py`, `home_page.py`, `main.py`.
   - `example_dataset/` - Small CSV examples used by the app:
     - `titanic.csv` (classification example)
     - `insurance.csv` (regression example)
   - `agent_module/` - Automation/agent code used for experiments and report generation. Contains two agent groups (`crewai` and `langchain`) with code, agent wrappers and summaries.
   - `sql_scripts/` - SQL example scripts for datasets (foldered by dataset name).
   - `logs/` - Application or agent logs (organized by date folders).
   - `test.ipynb`, `test.py` - small tests / notebooks used during development.
   - Other top-level docs: `R016_Report.docx`, `LICENSE`, `README.md`, `requirements.txt`.

   ## Features

   - Interactive EDA: summary statistics, charts and visualizations for quick inspection.
   - Data preprocessing: missing value handling, categorical encoding, scaling and outlier checks.
   - Feature engineering: PCA and K-best selection utilities.
   - Model training: basic classification and regression model training and evaluation using scikit-learn.
   - Sample datasets + custom dataset upload support.

   ## Supported models (examples)

   - Classification: Logistic Regression, Random Forest, SVM, Decision Tree, Gradient Boosting.
   - Regression: Linear Regression, Random Forest, Ridge, Lasso, SVM, Decision Tree, Gradient Boosting.

   ## Requirements

   - Python 3.8+ recommended. The project was developed with standard data science packages—see `requirements.txt` for the full list.

   ## Setup (Windows PowerShell)

   1. Create and activate a virtual environment (PowerShell):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

   3. Run the Streamlit app:

   ```powershell
   streamlit run src/main.py
   ```

   Streamlit will print a local URL (e.g. http://localhost:8501). Open that in your browser.

   Notes:
   - If PowerShell blocks script execution when activating the venv, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same session.

   ## How to use

   - Open the app and use the sidebar to pick a sample dataset or upload your own CSV/XLS file.
   - Work through the tabs: Custom EDA -> Data Preprocessing -> Feature Engineering -> Model Training.
   - Download or export trained models and evaluation summaries from the Model Training area when available.

   ## Development notes

   - The `agent_module/` folder contains prototype agents and helper scripts. These are experimental and primarily used to automate report generation and code-assisted EDA.
   - Logs are stored under `logs/YYYY-MM-DD/datacronyx.log` when run with logging enabled.

   ## Example: run one-off script

   - Run quick tests or scripts from the repo root (PowerShell):

   ```powershell
   python test.py
   ```

   ## Contributing

   - Bug reports and pull requests are welcome. Please include a short description of the environment and the steps to reproduce.

   ## License

   This project is available under the MIT License. See `LICENSE` for details.

   ## Acknowledgements

   - Streamlit, scikit-learn, pandas, seaborn, plotly and the open-source Python data ecosystem.

   ---

