```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    df = pd.read_csv('../example_dataset/titanic.csv')
except FileNotFoundError:
    print("Error: Titanic dataset not found. Place the file at the correct relative path.")
    exit()

# Basic information
info_buffer = []
info_buffer.append(f"Shape of the dataset: {df.shape}")
info_buffer.append(f"Data types of each column:\n{df.dtypes}")
info_buffer.append(f"Summary statistics:\n{df.describe()}")

# Missing values
info_buffer.append(f"Missing values:\n{df.isnull().sum()}")

# Impute missing Age values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Handle missing Embarked values by filling with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column due to too many missing values
df.drop('Cabin', axis=1, inplace=True)

info_buffer.append(f"Missing values after imputation:\n{df.isnull().sum()}")

# Outlier detection using boxplots for numerical features
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(x=df[col], ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.savefig('/code/boxplots.png')  # Save boxplots to a file

# Distribution of numerical features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df[col], ax=axes[i], kde=True)
    axes[i].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.savefig('/code/histograms.png') # Save histograms to a file

# Count plots for categorical features
cat_cols = ['Survived', 'Pclass', 'Sex', 'Embarked']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(x=df[col], ax=axes[i])
    axes[i].set_title(f'Countplot of {col}')

plt.tight_layout()
plt.savefig('/code/countplots.png') # Save countplots to a file

# Survival rate by Sex
survival_sex = df.groupby('Sex')['Survived'].mean()
info_buffer.append(f"Survival rate by Sex:\n{survival_sex}")

# Survival rate by Pclass
survival_pclass = df.groupby('Pclass')['Survived'].mean()
info_buffer.append(f"Survival rate by Pclass:\n{survival_pclass}")

# Save EDA information to a file
with open('/code/eda_info.txt', 'w') as f:
    for item in info_buffer:
        f.write(item + '\n\n')

print("EDA completed. Code saved to /code/eda_code.py. Visualizations saved to /code/boxplots.png, /code/histograms.png, and /code/countplots.png. EDA information saved to /code/eda_info.txt")
```