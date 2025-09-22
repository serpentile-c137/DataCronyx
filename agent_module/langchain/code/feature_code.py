import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("example_dataset/insurance.csv")

# Preprocessing (handle categorical variables and scale numerical features)
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
numerical_features = ['age', 'bmi', 'children', 'charges']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# 1. Create new features based on domain knowledge
data['bmi_age_interaction'] = data['bmi'] * data['age']
data['age_squared'] = data['age']**2
data['children_squared'] = data['children']**2
data['smoker_bmi_interaction'] = data['smoker_yes'] * data['bmi']

# 2. Feature interactions and polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_features = poly.fit_transform(data[['age', 'bmi', 'children']])
poly_df = pd.DataFrame(poly_features, columns = poly.get_feature_names_out(['age', 'bmi', 'children']))
data = pd.concat([data, poly_df], axis=1)

# 3. Feature selection techniques
X = data.drop('charges', axis=1)
y = data['charges']

# Univariate feature selection
selector = SelectKBest(score_func=f_regression, k=10)
selector.fit(X, y)
selected_features_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_features_indices]
X_selected = X[selected_features]

# 4. Dimensionality reduction (PCA)
pca = PCA(n_components=0.95) # Retain 95% of variance
X_pca = pca.fit_transform(X_selected)
pca_df = pd.DataFrame(X_pca, columns=[f'PCA{i+1}' for i in range(X_pca.shape[1])])

# 5. Feature importance analysis (using RandomForestRegressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
importances = model.feature_importances_
feature_importances = pd.Series(importances, index=X.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)

# Visualization of Feature Importances
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importances_sorted.head(10), y=feature_importances_sorted.head(10).index)
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Print selected features and PCA shape
print("Selected Features:", selected_features)
print("PCA Shape:", pca_df.shape)
print(feature_importances_sorted.head(10))