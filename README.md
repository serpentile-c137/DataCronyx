# DataCronyx

DataCronyx is an automated exploratory data analysis (EDA) and machine learning platform built with Streamlit. It provides an interactive interface for data exploration, preprocessing, feature engineering, and model training.

## Features

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
   ```bash
   streamlit run src/main.py
   ```

4. **Open in your browser:**  
   Streamlit will provide a local URL (e.g., http://localhost:8501).

## Usage

- Select a sample dataset or upload your own.
- Explore the data using the "Custom EDA" tab.
- Preprocess your data in the "Data Preprocessing" tab.
- Engineer features in the "Feature Engineering" tab.
- Train and evaluate models in the "Model Training" tab.

## Supported Models

### Classification

- Logistic Regression
- Random Forest
- SVM
- Decision Tree
- Gradient Boosting

### Regression

- Linear Regression
- Random Forest
- Ridge
- Lasso
- SVM
- Decision Tree
- Gradient Boosting

## File Structure

- `src/` - Source code for the Streamlit app and modules.
- `example_dataset/` - Sample datasets for demonstration.

## Requirements

- Python 3.7+
- See `requirements.txt` for Python package dependencies.

## License

MIT License

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [seaborn](https://seaborn.pydata.org/)
- [plotly](https://plotly.com/python/)

---

