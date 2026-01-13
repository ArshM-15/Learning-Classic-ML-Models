# House Price Prediction - Model Comparison

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("HOUSE PRICE PREDICTION - MODEL COMPARISON")
print("="*60)

# Load dataset
dataset = pd.read_csv('housing.csv')

print(f"\nTotal samples: {len(dataset)}")
print(f"Features: {dataset.columns.tolist()}\n")
print(dataset.head())

# Separate features and target
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# Encode binary columns
le = LabelEncoder()
for i in [4, 5, 6, 7, 8, 10]:
    X[:, i] = le.fit_transform(X[:, i])

# One-hot encode furnishingstatus
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [11])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(f"\nData encoded. New feature count: {X.shape[1]}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Feature scaling
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)

print("\n" + "="*60)
print("TRAINING MODELS...")
print("="*60)

results = {}

# Linear Regression
print("\nLinear Regression...")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
r2_lin = r2_score(y_test, y_pred_lin)
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
mae_lin = mean_absolute_error(y_test, y_pred_lin)
results['Linear Regression'] = {'R²': r2_lin, 'RMSE': rmse_lin, 'MAE': mae_lin}
print(f"R²: {r2_lin:.4f}, RMSE: ${rmse_lin:,.2f}, MAE: ${mae_lin:,.2f}")

# Polynomial Regression
print("\nPolynomial Regression (degree=2)...")
poly_reg = PolynomialFeatures(degree=2)
X_train_poly = poly_reg.fit_transform(X_train)
X_test_poly = poly_reg.transform(X_test)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_train_poly, y_train)
y_pred_poly = lin_reg_poly.predict(X_test_poly)
r2_poly = r2_score(y_test, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
mae_poly = mean_absolute_error(y_test, y_pred_poly)
results['Polynomial Regression'] = {
    'R²': r2_poly, 'RMSE': rmse_poly, 'MAE': mae_poly}
print(f"R²: {r2_poly:.4f}, RMSE: ${rmse_poly:,.2f}, MAE: ${mae_poly:,.2f}")

# Decision Tree
print("\nDecision Tree Regression...")
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
y_pred_tree = tree_reg.predict(X_test)
r2_tree = r2_score(y_test, y_pred_tree)
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
mae_tree = mean_absolute_error(y_test, y_pred_tree)
results['Decision Tree'] = {'R²': r2_tree, 'RMSE': rmse_tree, 'MAE': mae_tree}
print(f"R²: {r2_tree:.4f}, RMSE: ${rmse_tree:,.2f}, MAE: ${mae_tree:,.2f}")

# Random Forest
print("\nRandom Forest Regression...")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
results['Random Forest'] = {'R²': r2_rf, 'RMSE': rmse_rf, 'MAE': mae_rf}
print(f"R²: {r2_rf:.4f}, RMSE: ${rmse_rf:,.2f}, MAE: ${mae_rf:,.2f}")

# SVR
print("\nSVR...")
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_train_scaled, y_train)
y_pred_svr = svr_reg.predict(X_test_scaled)
r2_svr = r2_score(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
mae_svr = mean_absolute_error(y_test, y_pred_svr)
results['SVR'] = {'R²': r2_svr, 'RMSE': rmse_svr, 'MAE': mae_svr}
print(f"R²: {r2_svr:.4f}, RMSE: ${rmse_svr:,.2f}, MAE: ${mae_svr:,.2f}")

# Results summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"\n{'Model':<25} {'R² Score':<12} {'RMSE':<15} {'MAE':<15}")
print("-" * 70)

for model_name, metrics in results.items():
    print(
        f"{model_name:<25} {metrics['R²']:<12.4f} ${metrics['RMSE']:<14,.2f} ${metrics['MAE']:<14,.2f}")

# Best model
best_model = max(results.items(), key=lambda x: x[1]['R²'])
print("\n" + "="*60)
print(f"BEST MODEL: {best_model[0]}")
print(f"R² Score: {best_model[1]['R²']:.4f}")
print(f"RMSE: ${best_model[1]['RMSE']:,.2f}")
print(f"MAE: ${best_model[1]['MAE']:,.2f}")
print("="*60)
