import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("/var/folders/hn/z7dqkrys0jb521fxp_4sv30m0000gn/T/tmpjh1kms7r.csv")

# Display basic information
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nInfo:")
df.info()

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Check for duplicates
print("\nDuplicate Rows:", df.duplicated().sum())

# Statistical summary
print("\nStatistical Summary:\n", df.describe())

# Histograms
df.hist(figsize=(15, 10))
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Boxplots (detect outliers)
for column in df.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()

# Outlier identification using IQR
def find_outliers_iqr(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    outliers = data[((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr)))]
    return outliers

for column in df.select_dtypes(include=np.number).columns:
    outliers = find_outliers_iqr(df[column])
    print(f"\nOutliers in {column}:\n", outliers)

# Target variable analysis (assuming a target variable exists and is named 'target')
if 'target' in df.columns:
    print("\nTarget Variable Distribution:")
    print(df['target'].value_counts())
    sns.countplot(x='target', data=df)
    plt.title("Distribution of Target Variable")
    plt.show()