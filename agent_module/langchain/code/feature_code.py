import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the preprocessed data
data = pd.read_csv("/var/folders/hn/z7dqkrys0jb521fxp_4sv30m0000gn/T/tmp3k4x9s2m.csv")

# 1. Domain Knowledge Feature Engineering (Example - adjust based on actual data and domain)
data['feature_ratio_1'] = data['feature_1'] / (data['feature_2'] + 1e-6)  # Avoid division by zero
data['feature_product_1'] = data['feature_3'] * data['feature_4']
data['feature_diff_1'] = data['feature_5'] - data['feature_6']

# 2. Feature Interactions and Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_features = poly.fit_transform(data[['feature_1', 'feature_2', 'feature_3']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['feature_1', 'feature_2', 'feature_3']))
data = pd.concat([data, poly_df], axis=1)

# 3. Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

# 4. Feature Selection
X = scaled_df.drop('target', axis=1, errors='ignore')
y = scaled_df['target'] if 'target' in scaled_df.columns else pd.Series(np.random.rand(len(scaled_df))) # Dummy target if not present
selector = SelectKBest(score_func=f_regression, k=min(10, X.shape[1])) # Select top 10 or fewer
X_selected = selector.fit_transform(X, y)
selected_features_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_features_indices]

X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

# 5. Dimensionality Reduction (PCA)
pca = PCA(n_components=min(5, X_selected_df.shape[1]))  # Reduce to 5 components or fewer
pca_features = pca.fit_transform(X_selected_df)
pca_df = pd.DataFrame(pca_features, columns=[f'PCA_{i}' for i in range(pca_features.shape[1])])

# 6. Feature Importance Analysis (using RandomForestRegressor)
if 'target' in scaled_df.columns:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=X.columns)
    feature_importances_sorted = feature_importances.sort_values(ascending=False)

    # Plot top N important features
    N = min(10, len(feature_importances_sorted))
    plt.figure(figsize=(10, 6))
    feature_importances_sorted[:N].plot(kind='bar')
    plt.title('Top {} Feature Importances'.format(N))
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("Target variable not found, skipping feature importance analysis.")