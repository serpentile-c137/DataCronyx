```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import iqr

# Load the dataset
df = pd.read_csv("example_dataset/insurance.csv")

# 1. Handle missing values (if any) - Imputation
# No missing values were identified in EDA. Adding this for generalizability.
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent', 'constant'
df['age'] = imputer.fit_transform(df[['age']])
df['bmi'] = imputer.fit_transform(df[['bmi']])
df['children'] = imputer.fit_transform(df[['children']])
df['charges'] = imputer.fit_transform(df[['charges']])



# 2. Remove or treat outliers
# Outlier treatment using IQR method for 'bmi' and 'charges'
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    iqr_val = Q3 - Q1
    lower_bound = Q1 - 1.5 * iqr_val
    upper_bound = Q3 + 1.5 * iqr_val
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

df = remove_outliers_iqr(df, 'bmi')
df = remove_outliers_iqr(df, 'charges')


# 3. Encode categorical variables
# Label encode 'sex', 'smoker', and 'region'
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])

# One-hot encode 'region'
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# 4. Scale numerical features (Standard Scaling)
numerical_features = ['age', 'bmi', 'children', 'charges']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# 5. Create train-test split
X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Save preprocessed data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
```