import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('../example_dataset/titanic.csv')

# 2. Display basic information
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nInfo:")
df.info()

# 3. Check for missing values and duplicates
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# 4. Generate statistical summaries
print("\nStatistical Summary:\n", df.describe())
print("\nStatistical Summary (Categorical):\n", df.describe(include=['object']))

# 5. Create visualizations
# Histograms
df.hist(figsize=(12, 10))
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Boxplots
numerical_features = df.select_dtypes(include=np.number).columns
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot of {feature}")
    plt.show()

# 6. Identify outliers (using IQR)
def identify_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

for feature in numerical_features:
    outliers = identify_outliers(df, feature)
    print(f"\nOutliers in {feature}:\n", outliers)

# 7. Analyze target variable distribution (if applicable)
if 'Survived' in df.columns:
    print("\nTarget Variable Distribution (Survived):\n", df['Survived'].value_counts())
    sns.countplot(x='Survived', data=df)
    plt.title("Distribution of Survived")
    plt.show()

# Additional Visualizations

# Survival rate by Sex
if 'Survived' in df.columns and 'Sex' in df.columns:
    sns.countplot(x='Sex', hue='Survived', data=df)
    plt.title('Survival Rate by Sex')
    plt.show()

# Survival rate by Pclass
if 'Survived' in df.columns and 'Pclass' in df.columns:
    sns.countplot(x='Pclass', hue='Survived', data=df)
    plt.title('Survival Rate by Pclass')
    plt.show()

# Survival rate by Embarked
if 'Survived' in df.columns and 'Embarked' in df.columns:
    sns.countplot(x='Embarked', hue='Survived', data=df)
    plt.title('Survival Rate by Embarked')
    plt.show()

# Pairplot for numerical features
sns.pairplot(df[numerical_features.tolist()])
plt.suptitle("Pairplot of Numerical Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()