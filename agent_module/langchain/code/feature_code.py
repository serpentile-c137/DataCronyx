import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("/var/folders/hn/z7dqkrys0jb521fxp_4sv30m0000gn/T/tmpjh1kms7r.csv")

# 1. Domain Knowledge Based Features (Example: Assuming features related to time or location)
# Example 1: Time-based feature (assuming a 'timestamp' column exists)
#df['timestamp'] = pd.to_datetime(df['timestamp'])
#df['hour_of_day'] = df['timestamp'].dt.hour
#df['day_of_week'] = df['timestamp'].dt.dayofweek

# Example 2: Location-based feature (assuming 'latitude' and 'longitude' columns exist)
#df['lat_lon_interaction'] = df['latitude'] * df['longitude']

# 2. Feature Interactions and Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_features = poly.fit_transform(df)
poly_feature_names = poly.get_feature_names_out(input_features=df.columns)
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)

# 3. Feature Selection
X = df_poly  # Using polynomial features
y = df['target'] #Assuming 'target' column exists

# Univariate Feature Selection
selector = SelectKBest(score_func=f_regression, k=10)
selector.fit(X, y)
selected_features_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_features_indices]
X_selected = X[selected_features]

# 4. Dimensionality Reduction (PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

pca = PCA(n_components=0.95)  # Retain 95% of variance
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca)

# 5. Feature Importance Analysis (Using Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_selected, y)
importances = model.feature_importances_

feature_importances = pd.DataFrame({'Feature': X_selected.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values('Importance', ascending=False)

# Print or visualize feature importances
#print(feature_importances)

# Example Visualization
#plt.figure(figsize=(10, 6))
#plt.bar(feature_importances['Feature'], feature_importances['Importance'])
#plt.xticks(rotation=45, ha='right')
#plt.xlabel('Feature')
#plt.ylabel('Importance')
#plt.title('Feature Importances')
#plt.tight_layout()
#plt.show()

# Final Dataframe (PCA applied to selected Polynomial Features)
final_df = pd.concat([df_pca, y], axis=1)