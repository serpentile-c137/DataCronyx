import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pickle

# Load the dataset
df = pd.read_csv('/var/folders/hn/z7dqkrys0jb521fxp_4sv30m0000gn/T/tmpjh1kms7r.csv')

# 1. Handle missing values
# Impute numerical missing values with the mean
numerical_cols = df.select_dtypes(include=np.number).columns
imputer_numerical = SimpleImputer(strategy='mean')
df[numerical_cols] = imputer_numerical.fit_transform(df[numerical_cols])

# Impute categorical missing values with the mode
categorical_cols = df.select_dtypes(exclude=np.number).columns
imputer_categorical = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])

# 2. Remove or treat outliers
# Remove outliers based on IQR for numerical columns
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# 3. Encode categorical variables
# Label encode categorical features
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# 4. Scale numerical features
# Standardize numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 5. Create train-test split
X = df.drop(columns=['target'], errors='ignore')  # Replace 'target' with actual target column name if it exists
y = df['target'] if 'target' in df.columns else None  # Replace 'target' with actual target column name if it exists

if y is not None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    y_train, y_test = None, None

# 6. Save preprocessed data and fitted objects
X_train.to_csv('X_train_preprocessed.csv', index=False)
X_test.to_csv('X_test_preprocessed.csv', index=False)

if y_train is not None and y_test is not None:
    pd.Series(y_train).to_csv('y_train.csv', index=False)
    pd.Series(y_test).to_csv('y_test.csv', index=False)

with open('numerical_imputer.pkl', 'wb') as file:
    pickle.dump(imputer_numerical, file)

with open('categorical_imputer.pkl', 'wb') as file:
    pickle.dump(imputer_categorical, file)

with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)