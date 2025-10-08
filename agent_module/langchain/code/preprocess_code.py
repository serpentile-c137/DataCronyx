import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

# Load the dataset
df = pd.read_csv('example_dataset/titanic.csv')

# Handle missing values
# Impute age with median
imputer_age = SimpleImputer(strategy='median')
df['Age'] = imputer_age.fit_transform(df[['Age']])

# Impute Embarked with mode
imputer_embarked = SimpleImputer(strategy='most_frequent')
df['Embarked'] = imputer_embarked.fit_transform(df[['Embarked']])

# Fill Cabin with 'Unknown' category
df['Cabin'] = df['Cabin'].fillna('Unknown')

# Remove outliers (example: based on EDA, remove high Fare values)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[df['Fare'] <= (Q3 + 1.5 * IQR)]

# Encode categorical variables
# Encode Sex
label_encoder_sex = LabelEncoder()
df['Sex'] = label_encoder_sex.fit_transform(df['Sex'])

# Encode Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Encode Cabin (simplified - first letter only)
df['Cabin'] = df['Cabin'].astype(str).str[0]
df = pd.get_dummies(df, columns=['Cabin'], drop_first=True)

# Drop unnecessary columns
df = df.drop(['Name', 'Ticket'], axis=1)

# Scale numerical features
numerical_features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Prepare data for train-test split
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Save scalers and encoders
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder_sex, 'label_encoder_sex.joblib')
joblib.dump(imputer_age, 'imputer_age.joblib')
joblib.dump(imputer_embarked, 'imputer_embarked.joblib')