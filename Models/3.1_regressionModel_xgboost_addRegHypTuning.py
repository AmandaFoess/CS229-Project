import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import os

# Load the dataset
# Ensure the file exists and contains the correct data format
df = pd.read_csv('(3) TransformedData/Addresses_with_Ratios_and_New_Features.csv')
print("Dataset loaded successfully.")

# Handle missing values
# Drop rows where any of the key features are missing
df.dropna(subset=['Acres', 'VDL Sale Price', 'Latitude', 'Longitude', 'landuseful'], inplace=True)
print(f"After dropping rows with missing values, dataset has {len(df)} records.")

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
print(f"After removing outliers, dataset has {len(df)} records.")

# Extract year from saledate
# Convert saledate to datetime format and extract the year
df['saledate'] = pd.to_datetime(df['saledate'])
df['sale_year'] = df['saledate'].dt.year
print("Sale year extracted successfully.")

# Define features and target variable
features = ['Acres', 'VDL Sale Price', 'Latitude', 'Longitude', 'sale_year', 'landuseful']
X = df[features]
y = df['House to Lot Ratio']
print(f"Features and target variable defined. Number of features: {len(features)}.")

# Preprocess the data
# Separate numeric and categorical features
numeric_features = ['Acres', 'VDL Sale Price', 'Latitude', 'Longitude', 'sale_year']
categorical_features = ['landuseful']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Encode categorical features
    ])
print("Preprocessor created successfully.")

# Create the XGBoost model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42))
])
print("Model pipeline created successfully.")

# Split the data
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into training and testing sets. Training set: {len(X_train)} records, Test set: {len(X_test)} records.")

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [100, 300, 500],
    'regressor__max_depth': [4, 6, 8],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__subsample': [0.6, 0.8, 1.0],
    'regressor__colsample_bytree': [0.6, 0.8, 1.0],
    'regressor__reg_alpha': [0, 0.1, 1],  # L1 regularization
    'regressor__reg_lambda': [1, 10, 100]  # L2 regularization
}
print("Parameter grid defined for hyperparameter tuning.")

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', verbose=1, n_jobs=-1)
print("Starting grid search for hyperparameter tuning...")
grid_search.fit(X_train, y_train)
print("Grid search completed.")

# Get the best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Make predictions with the best model
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
print("Predictions made with the best model.")

# Evaluate the Model
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Training MSE: {mse_train:.2f}")
print(f"Test MSE: {mse_test:.2f}")
print(f"Training R^2: {r2_train:.2f}")
print(f"Test R^2: {r2_test:.2f}")

# Plot Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6, label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label="Ideal Fit (y = x)")
plt.title('Predicted vs Actual House-to-Lot Ratio (XGBoost with Best Params)')
plt.xlabel('Actual House-to-Lot Ratio')
plt.ylabel('Predicted House-to-Lot Ratio')
plt.legend()
plt.tight_layout()

# Save the plot
output_dir = "Models/Outputs"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "3.1xgboost_predicted_vs_actual_cv5.png")
plt.savefig(plot_path)
plt.show()
print(f"Plot saved at {plot_path}.")