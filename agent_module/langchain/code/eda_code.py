import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('/var/folders/hn/z7dqkrys0jb521fxp_4sv30m0000gn/T/tmp3k4x9s2m.csv')

# 2. Display basic information
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nInfo:\n", df.info())

# 3. Check for missing values and duplicates
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicated rows:", df.duplicated().sum())

# Remove duplicates if any
df.drop_duplicates(inplace=True)

# 4. Generate statistical summaries
print("\nStatistical summary:\n", df.describe())

# 5. Create visualizations
# Histograms for numerical features
df.hist(figsize=(12, 10))
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Boxplots for numerical features
for column in df.select_dtypes(include=np.number):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()

# 6. Identify outliers (using IQR method)
def detect_outliers_iqr(data):
    outliers = {}
    for col in data.select_dtypes(include=np.number):
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col].index.tolist()
    return outliers

outliers = detect_outliers_iqr(df)
print("\nOutliers (IQR method):\n", outliers)

# 7. Analyze target variable distribution (if applicable)
# Assuming there is a target variable named 'target'
if 'target' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df['target'], kde=True)
    plt.title("Distribution of Target Variable")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['target'])
    plt.title("Boxplot of Target Variable")
    plt.show()
else:
    print("\nNo 'target' column found. Skipping target variable analysis.")