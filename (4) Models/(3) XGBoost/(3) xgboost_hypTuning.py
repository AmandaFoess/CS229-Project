import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import os

# Load the dataset
input_csv = "(3) TransformedData/Addresses_with_Ratio.csv"
df = pd.read_csv(input_csv)

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

# Preprocess features
numeric_features = ['Acres', 'VDL Sale Price', 'Finished Home Value', 'Latitude', 'Longitude', 'sale_year']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'regressor__n_estimators': [100, 200, 500],  # Number of boosting rounds
    'regressor__max_depth': [4, 6, 8],          # Maximum depth of trees
    'regressor__learning_rate': [0.01, 0.05, 0.1],  # Shrinkage for updates
    'regressor__subsample': [0.6, 0.8, 1.0],    # Fraction of samples used per tree
    'regressor__colsample_bytree': [0.6, 0.8, 1.0]  # Fraction of features used per tree
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=param_grid,
    scoring='r2',  # Use R^2 as the evaluation metric
    cv=5,          # 5-fold cross-validation
    n_jobs=-1,     # Use all available CPU cores
    verbose=2      # Print progress during the search
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validated R^2: {grid_search.best_score_:.2f}")

# Evaluate the model on the test set
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)

mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Test MSE with Best Parameters: {mse_test:.2f}")
print(f"Test R^2 with Best Parameters: {r2_test:.2f}")

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
plot_path = os.path.join(output_dir, "xgb_best_predicted_vs_actual.png")
plt.savefig(plot_path)
plt.show()
