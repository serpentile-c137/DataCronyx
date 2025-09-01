# DataCronyx

**DataCronyx** is an interactive web application for Automated Exploratory Data Analysis (AutoEDA) and data preprocessing. It provides a user-friendly interface for data analysts, scientists, business professionals, and students to explore, visualize, and preprocess datasets with minimal coding.

## Features

- **Interactive Data Exploration:** Visualize and analyze datasets with summary statistics, missing value reports, and data type overviews.
- **Custom EDA:** Generate histograms, scatter plots, box plots, and correlation heatmaps for in-depth data understanding.
- **Data Preprocessing:** Remove columns, handle missing values, encode categorical variables, scale features, and detect/handle outliers.
- **One-click Download:** Export your preprocessed data for further analysis or modeling.
- **User-Friendly Interface:** Built with [Streamlit](https://streamlit.io/) for an intuitive and responsive experience.

## Getting Started

### Prerequisites

- Python 3.7+
- Recommended: Create a virtual environment

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/serpentile-c137/DataCronyx.git
   cd DataCronyx
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, install: `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `plotly`)*

### Running the App

```bash
streamlit run src/main.py
```

Open your browser at [http://localhost:8501](http://localhost:8501) to use DataCronyx.

## Usage

- **Home:** Overview and instructions.
- **Custom EDA:** Upload your CSV or use the example Titanic dataset. Explore data structure, missing values, statistics, and visualizations.
- **Data Preprocessing:** Remove columns, handle missing data, encode categorical features, scale numerical features, and manage outliers. Download the cleaned dataset.

## File Structure

- `src/main.py` - Main Streamlit app entry point.
- `src/data_analysis_functions.py` - Functions for EDA and visualization.
- `src/data_preprocessing_function.py` - Functions for data cleaning and preprocessing.
- `src/home_page.py` - Home page UI and custom CSS.
- `example_dataset/` - Example datasets (e.g., Titanic).

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT License](LICENSE)

## Acknowledgements

- Built with [Streamlit](https://streamlit.io/)
- Visualization powered by [Plotly](https://plotly.com/) and [Seaborn](https://seaborn.pydata.org/)

---
*Unleash the Power of Data with DataCronyx!*

