import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("example_dataset/insurance.csv")

# 2. Display basic information
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nInfo:")
df.info()

# 3. Check for missing values and duplicates
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# Remove duplicates if any
df.drop_duplicates(inplace=True)

# 4. Generate statistical summaries
print("\nStatistical Summary:\n", df.describe())
print("\nStatistical Summary (Categorical):\n", df.describe(include=['object']))

# 5. Create visualizations

# Histograms
df.hist(figsize=(12, 10))
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
plt.show()

# Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Boxplots
numerical_features = df.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(15, 8))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, len(numerical_features) // 2 + 1, i + 1)
    sns.boxplot(y=df[feature])
    plt.title(f"Boxplot of {feature}")
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(df)
plt.suptitle("Pairplot of Numerical Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Analyze categorical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=feature, data=df)
    plt.title(f"Countplot of {feature}")
    plt.show()

# 6. Identify outliers (using IQR method)
def detect_outliers_iqr(data):
    outliers = []
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    for x in data:
        if (x < lower_bound) or (x > upper_bound):
            outliers.append(x)
    return outliers

for feature in numerical_features:
    outliers = detect_outliers_iqr(df[feature])
    print(f"\nOutliers in {feature}: {len(outliers)}")

# 7. Analyze target variable distribution (charges)
plt.figure(figsize=(8, 6))
sns.histplot(df['charges'], kde=True)
plt.title("Distribution of Charges")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(y=df['charges'])
plt.title("Boxplot of Charges")
plt.show()

# Additional analysis: Charges vs. other features

# Charges vs. Smoker
plt.figure(figsize=(8, 6))
sns.boxplot(x='smoker', y='charges', data=df)
plt.title("Charges vs. Smoker")
plt.show()

# Charges vs. Region
plt.figure(figsize=(8, 6))
sns.boxplot(x='region', y='charges', data=df)
plt.title("Charges vs. Region")
plt.show()

# Charges vs. Age
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='charges', data=df)
plt.title("Charges vs. Age")
plt.show()

# Charges vs. BMI
plt.figure(figsize=(8, 6))
sns.scatterplot(x='bmi', y='charges', data=df)
plt.title("Charges vs. BMI")
plt.show()

# Charges vs. Children
plt.figure(figsize=(8, 6))
sns.boxplot(x='children', y='charges', data=df)
plt.title("Charges vs. Children")
plt.show()