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
df = pd.read_csv('(3) TransformedData/Addresses_with_Mapped_Grantees_Cleaned.csv')

# Convert 'saledate' to numeric representation
df['saledate'] = pd.to_datetime(df['saledate'], format='%Y-%m')
df['saledate_numeric'] = (df['saledate'].dt.year - df['saledate'].dt.year.min()) * 12 + df['saledate'].dt.month
df.drop(columns=['saledate'], inplace=True)

# Define features and target variable
features = ['Acres', 'VDL Sale Price', 'Latitude', 'Longitude', 'saledate_numeric', 'landuseful', 'Mapped_Grantee']
X = df[features]
y = df['House to Lot Ratio']

# Preprocess the data
numeric_features = ['Acres', 'VDL Sale Price', 'Latitude', 'Longitude', 'saledate_numeric']
categorical_features = ['landuseful', 'Mapped_Grantee']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Encode categorical features
    ])

# Create the XGBoost model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [100, 300, 500],
    'regressor__max_depth': [8],
    'regressor__learning_rate': [0.1],
    'regressor__subsample': [1.0],
    'regressor__colsample_bytree': [0.6],
    'regressor__reg_alpha': [0, 0.05, 0.1, 0.15, 0.2],  # L1 regularization
    'regressor__reg_lambda': [0.4, 0.5, 0.6]  # L2 regularization
}

# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Evaluate the Model
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Best Parameters: {grid_search.best_params_}")
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
plot_path = os.path.join(output_dir, "xgboost_predicted_vs_actual.png")
plt.savefig(plot_path)
plt.show()