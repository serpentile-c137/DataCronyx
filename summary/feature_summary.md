```markdown
# Titanic Dataset Feature Engineering Rationale

This document outlines the feature engineering steps applied to the Titanic dataset to improve the performance of machine learning models.

## 1. Feature: FamilySize

- **Rationale:**  Combining `SibSp` (number of siblings/spouses aboard) and `Parch` (number of parents/children aboard) can provide a more comprehensive measure of family size.  A larger family might influence survival chances.
- **Implementation:**
  ```python
  df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
  ```

## 2. Feature: IsAlone

- **Rationale:**  Individuals traveling alone might have different survival characteristics compared to those with family members. This binary feature captures this information.
- **Implementation:**
  ```python
  df['IsAlone'] = df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
  ```

## 3. Feature: Title

- **Rationale:**  The title of a passenger (e.g., Mr., Mrs., Miss., Master) can indicate social status and age, which might correlate with survival rate.
- **Implementation:**
  ```python
  df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
  df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
  df['Title'] = df['Title'].replace('Mlle', 'Miss')
  df['Title'] = df['Title'].replace('Ms', 'Miss')
  df['Title'] = df['Title'].replace('Mme', 'Mrs')
  ```
  - The code extracts titles from the `Name` column using regular expressions.
  - Rare titles are grouped into a single 'Rare' category to reduce dimensionality and avoid overfitting.
  - Titles like 'Mlle', 'Ms', and 'Mme' are consolidated into their more common equivalents ('Miss' and 'Mrs').

## 4. Feature: Age Category

- **Rationale:**  Categorizing age into bins can help capture non-linear relationships between age and survival.  Different age groups might have had different priorities during the evacuation.
- **Implementation:**
  ```python
  df['Age_Category'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, np.inf], labels=['Child', 'Teenager', 'Adult', 'Senior'])
  ```
  - The `Age` is divided into categories: 'Child', 'Teenager', 'Adult', and 'Senior'.

## 5. Feature: Fare Category

- **Rationale:** Similar to age, fare might have a non-linear relationship with survival.  Discretizing fare into categories can help capture these relationships.
- **Implementation:**
  ```python
  df['Fare_Category'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
  ```
  - The `Fare` is divided into four quantiles: 'Low', 'Medium', 'High', and 'Very High'.

## Integration with Preprocessing Pipeline

These engineered features are then integrated into the preprocessing pipeline, which includes:

- **Numerical Feature Transformation:**
  - Imputation of missing values using the median.
  - Scaling using `StandardScaler`.
- **Categorical Feature Transformation:**
  - Imputation of missing values using the most frequent value.
  - One-hot encoding using `OneHotEncoder`.

The complete `preprocess_titanic_data` function, including feature engineering, is shown below:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

def engineer_features(df):
    """
    Engineers new features from the Titanic dataset.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new features.
    """

    # 1. Feature: FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 2. Feature: IsAlone
    df['IsAlone'] = df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)

    # 3. Feature: Title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # 4. Feature: Age Category
    df['Age_Category'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, np.inf], labels=['Child', 'Teenager', 'Adult', 'Senior'])

    # 5. Feature: Fare Category
    df['Fare_Category'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

    return df


def preprocess_titanic_data(file_path, test_size=0.2, random_state=42):
    """
    Preprocesses the Titanic dataset, including feature engineering.

    Args:
        file_path (str): The path to the Titanic dataset CSV file.
        test_size (float): The proportion of the dataset to use for testing.
        random_state (int): The random state for splitting the data.

    Returns:
        tuple: A tuple containing the training data (X_train, y_train) and
               the testing data (X_test, y_test) as pandas DataFrames.
    """

    # Load the dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError("Titanic dataset not found. Please ensure the file path is correct.")

    # Separate features (X) and target (y)
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Engineer features
    X = engineer_features(X)

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns  # Include 'category' type

    # Create a column transformer to apply different transformations to different columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any other columns not specified
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Preprocess the data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Convert the processed data back to pandas DataFrames
    X_train = pd.DataFrame(X_train, columns=preprocessor.get_feature_names_out())
    X_test = pd.DataFrame(X_test, columns=preprocessor.get_feature_names_out())

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, y_train, X_test, y_test
```

By engineering these features, we aim to provide the machine learning model with more relevant and informative data, potentially leading to improved prediction accuracy.
```