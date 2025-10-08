import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("example_dataset/titanic.csv")

# 2. Display basic information
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nInfo:")
df.info()

# 3. Check for missing values and duplicates
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# 4. Generate statistical summaries
print("\nStatistical Summary:\n", df.describe())
print("\nStatistical Summary (Categorical):\n", df.describe(include=['O']))

# 5. Create visualizations
# Histograms
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

# Boxplots
numerical_features = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(15, 8))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 4, i + 1)
    sns.boxplot(y=df[feature])
    plt.title(f"Boxplot of {feature}")
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(df)
plt.suptitle("Pairplot of Numerical Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 6. Identify outliers (using IQR method)
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

for feature in numerical_features:
    outliers = detect_outliers(df, feature)
    print(f"\nOutliers in {feature}:\n", outliers)

# 7. Analyze target variable distribution (Survived)
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title("Distribution of Survival")
plt.show()

# Survival rate by Sex
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Sex")
plt.show()

# Survival rate by Pclass
plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Pclass")
plt.show()