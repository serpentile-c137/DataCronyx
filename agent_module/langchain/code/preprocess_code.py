import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

# Load the dataset
df = pd.read_csv("../example_dataset/titanic.csv")

# Handle missing values
# Impute 'Age' with the median
df["Age"].fillna(df["Age"].median(), inplace=True)
# Impute 'Embarked' with the mode
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
# 'Cabin' has too many missing values, so drop it
df.drop("Cabin", axis=1, inplace=True)


# Treat outliers (using IQR method for 'Fare' and 'Age')
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[
        (df[column] >= lower_bound) & (df[column] <= upper_bound)
    ]  # Keep rows within bounds
    return df


df = remove_outliers_iqr(df, "Fare")
df = remove_outliers_iqr(df, "Age")

# Encode categorical variables
# Use LabelEncoder for 'Sex'
label_encoder = LabelEncoder()
df["Sex"] = label_encoder.fit_transform(df["Sex"])

# Use OneHotEncoder for 'Embarked'
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_embarked = onehot_encoder.fit_transform(df[["Embarked"]])
embarked_df = pd.DataFrame(
    encoded_embarked, columns=onehot_encoder.get_feature_names_out(["Embarked"])
)
df = pd.concat([df.reset_index(drop=True), embarked_df.reset_index(drop=True)], axis=1)
df.drop("Embarked", axis=1, inplace=True)  # Drop original 'Embarked' column


# Drop irrelevant features
df.drop(["Name", "Ticket"], axis=1, inplace=True)


# Scale numerical features ('Age', 'Fare', 'Pclass', 'SibSp', 'Parch')
numerical_features = ["Age", "Fare", "Pclass", "SibSp", "Parch"]
scaler = StandardScaler()  # or MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Train-test split
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save preprocessed data
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)