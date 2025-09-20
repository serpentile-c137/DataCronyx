import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('../example_dataset/titanic.csv')

# Preprocessing (handle missing values and categorical variables)
# Impute missing Age values with the median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Impute missing Embarked values with the mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

# Drop unnecessary columns
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Feature Engineering
# 1. Create new features based on domain knowledge
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)
df['FarePerPerson'] = df['Fare'] / df['FamilySize']
df['FarePerPerson'] = df['FarePerPerson'].replace([np.inf, -np.inf], 0) #Handle division by zero

# 2. Feature interactions and polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(df[['Age', 'Fare', 'Pclass']])
poly_df = pd.DataFrame(poly_features, columns = poly.get_feature_names_out(['Age', 'Fare', 'Pclass']))
df = pd.concat([df.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)

# 3. Feature Scaling
numerical_features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'FamilySize', 'FarePerPerson'] + list(poly_df.columns)
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Prepare data for feature selection
X = df.drop('Survived', axis=1)
y = df['Survived']

# 4. Feature Selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_features_indices]
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)


# 5. Dimensionality Reduction (PCA) - only if number of features is high
if X_selected_df.shape[1] > 15:
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_selected_df)
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(10)])
    X_final = X_pca_df
else:
    X_final = X_selected_df

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 6. Feature Importance Analysis
if 'X_pca_df' not in locals():
    importances = model.feature_importances_
    feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
    feature_importances = feature_importances.sort_values('Importance', ascending=False)
    print("\nFeature Importances:")
    print(feature_importances)