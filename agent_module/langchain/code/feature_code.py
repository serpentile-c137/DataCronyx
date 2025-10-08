import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('example_dataset/titanic.csv')

# Preprocessing (handling missing values and encoding categorical features)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Drop unnecessary columns
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Feature Engineering
# 1. Family Size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 2. IsAlone
df['IsAlone'] = np.where(df['FamilySize'] > 1, 0, 1)

# 3. Age Group
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teenager', 'Adult', 'Senior'])
df['AgeGroup'] = df['AgeGroup'].map({'Child': 0, 'Teenager': 1, 'Adult': 2, 'Senior': 3})

# Feature Interactions and Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_features = poly.fit_transform(df[['Age', 'Fare', 'Pclass']])
poly_df = pd.DataFrame(poly_features, columns = poly.get_feature_names_out(['Age', 'Fare', 'Pclass']))
df = pd.concat([df.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)

# Feature Scaling
numerical_features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'FamilySize']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Feature Selection
X = df.drop('Survived', axis=1)
y = df['Survived']

selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(X, y)

selected_features = X.columns[selector.get_support()]
X_selected = X[selected_features]

# Dimensionality Reduction (PCA)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_selected)

# Feature Importance Analysis (using RandomForest)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

importances = model.feature_importances_
feature_importances = pd.Series(importances, index=X.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances_sorted.plot(kind='bar')
plt.title('Feature Importances')
plt.show()

print(feature_importances_sorted)