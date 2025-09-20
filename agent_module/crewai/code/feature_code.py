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


if __name__ == "__main__":
    # Example usage:
    try:
        X_train, y_train, X_test, y_test = preprocess_titanic_data('../example_dataset/titanic.csv')

        print("Shape of X_train:", X_train.shape)
        print("Shape of X_test:", X_test.shape)
        print("Shape of y_train:", y_train.shape)
        print("Shape of y_test:", y_test.shape)

        print("\nFirst 5 rows of X_train:")
        print(X_train.head())

        print("\nFirst 5 rows of X_test:")
        print(X_test.head())

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
```