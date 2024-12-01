import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Load the dataset
df = pd.read_csv('(3) TransformedData/Addresses_with_Ratio.csv')

# Handle missing values
df.dropna(subset=['Acres', 'VDL Sale Price', 'Finished Home Value', 'Latitude', 'Longitude'], inplace=True)

# Outlier removal using IQR
def remove_outliers(df, column, quantile_range):
    """Removes outliers based on the IQR method."""
    Q1 = df[column].quantile(quantile_range)
    Q3 = df[column].quantile(1 - quantile_range)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from relevant columns
df = remove_outliers(df, "House to Lot Ratio", 0.25)
df = remove_outliers(df, "Longitude", 0.05)
df = remove_outliers(df, "Latitude", 0.05)

# Extract year from saledate
df['saledate'] = pd.to_datetime(df['saledate'])
df['sale_year'] = df['saledate'].dt.year

# Define features and target variable
features = ['Acres', 'VDL Sale Price', 'Finished Home Value', 'Latitude', 'Longitude', 'sale_year']
X = df[features]
y = df['House to Lot Ratio']

# Preprocess numeric features
numeric_features = ['Acres', 'VDL Sale Price', 'Finished Home Value', 'Latitude', 'Longitude', 'sale_year']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipelines for Ridge and Lasso
ridge_model = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', Ridge(alpha=1.0))])  # Ridge Regression with alpha=1.0

lasso_model = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', Lasso(alpha=0.01, max_iter=10000))])  # Lasso Regression with alpha=0.01

# Train Ridge Regression
ridge_model.fit(X_train, y_train)
y_train_pred_ridge = ridge_model.predict(X_train)
y_test_pred_ridge = ridge_model.predict(X_test)

# Train Lasso Regression
lasso_model.fit(X_train, y_train)
y_train_pred_lasso = lasso_model.predict(X_train)
y_test_pred_lasso = lasso_model.predict(X_test)

# Evaluate Ridge Regression
print("\nRidge Regression:")
mse_train_ridge = mean_squared_error(y_train, y_train_pred_ridge)
mse_test_ridge = mean_squared_error(y_test, y_test_pred_ridge)
r2_train_ridge = r2_score(y_train, y_train_pred_ridge)
r2_test_ridge = r2_score(y_test, y_test_pred_ridge)

print(f"Training MSE: {mse_train_ridge:.2f}")
print(f"Test MSE: {mse_test_ridge:.2f}")
print(f"Training R^2: {r2_train_ridge:.2f}")
print(f"Test R^2: {r2_test_ridge:.2f}")

# Evaluate Lasso Regression
print("\nLasso Regression:")
mse_train_lasso = mean_squared_error(y_train, y_train_pred_lasso)
mse_test_lasso = mean_squared_error(y_test, y_test_pred_lasso)
r2_train_lasso = r2_score(y_train, y_train_pred_lasso)
r2_test_lasso = r2_score(y_test, y_test_pred_lasso)

print(f"Training MSE: {mse_train_lasso:.2f}")
print(f"Test MSE: {mse_test_lasso:.2f}")
print(f"Training R^2: {r2_train_lasso:.2f}")
print(f"Test R^2: {r2_test_lasso:.2f}")

# Plot Predicted vs Actual for Ridge Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred_ridge, alpha=0.6, label="Ridge Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label="Ideal Fit (y = x)")
plt.title('Predicted vs Actual House-to-Lot Ratio (Ridge)')
plt.xlabel('Actual House-to-Lot Ratio')
plt.ylabel('Predicted House-to-Lot Ratio')
plt.legend()
plt.tight_layout()

# Save Ridge plot
output_dir = "Models/Outputs"
os.makedirs(output_dir, exist_ok=True)
ridge_plot_path = os.path.join(output_dir, "ridge_predicted_vs_actual.png")
plt.savefig(ridge_plot_path)
plt.show()

# Plot Predicted vs Actual for Lasso Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred_lasso, alpha=0.6, label="Lasso Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label="Ideal Fit (y = x)")
plt.title('Predicted vs Actual House-to-Lot Ratio (Lasso)')
plt.xlabel('Actual House-to-Lot Ratio')
plt.ylabel('Predicted House-to-Lot Ratio')
plt.legend()
plt.tight_layout()

# Save Lasso plot
lasso_plot_path = os.path.join(output_dir, "lasso_predicted_vs_actual.png")
plt.savefig(lasso_plot_path)
plt.show()
