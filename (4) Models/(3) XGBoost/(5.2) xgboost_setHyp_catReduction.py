import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import os

# Load the dataset
df = pd.read_csv('(3) TransformedData/Addresses_with_Ratios_and_New_Features.csv')

# Handle missing values
df.dropna(subset=['Acres', 'VDL Sale Price', 'Latitude', 'Longitude', 'landuseful'], inplace=True)

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

# Step 1: Map landuse features to broader categories
landuse_category_mapping = {
    "USE VALUE HOMESITE": "single family",
    "RURAL HOMESITE": "single family",
    "SINGLE FAMILY RESIDENTIAL - ACREAGE": "single family",
    "SINGLE FAMILY RESIDENTIAL - GOLF": "single family",
    "SINGLE FAMILY RESIDENTIAL": "single family",
    "SINGLE FAMILY RESIDENTIAL - COMMON": "single family",
    "SINGLE FAMILY RESIDENTIAL - WATERFRONT": "single family",
    "SINGLE FAMILY RESIDENTIAL - RIVER": "single family",
    "TOWN HOUSE SFR": "townhomes",
    "MULTI FAMILY": "multifamily",
    "MULTI FAMILY GARDEN": "multifamily",
    "MULTI FAMILY DUPLEX/TRIPLEX": "multifamily",
    "MULTI FAMILY MARINA LAND": "multifamily"
}

# Step 2: Map landuseful to categories and drop unmatched rows
df['landuse_category'] = df['landuseful'].map(landuse_category_mapping)

# Remove rows where landuse_category is NaN (not in mapping)
df = df.dropna(subset=['landuse_category'])

# Step 3: Replace the old landuseful column with the new categories (optional)
df['landuseful'] = df['landuse_category']
df = df.drop(columns=['landuse_category'])  # Drop temporary column if no longer needed

# Step 2: Update feature lists
numeric_features = ['Acres', 'VDL Sale Price', 'Latitude', 'Longitude', 'sale_year']
categorical_features = ['landuseful']  # Ensure only important categories remain

# Define features and target variable
features = numeric_features + categorical_features
X = df[features]
y = df['House to Lot Ratio']

# Preprocess the data
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
    'regressor__n_estimators': [300],
    'regressor__max_depth': [8],
    'regressor__learning_rate': [0.1],
    'regressor__subsample': [1.0],
    'regressor__colsample_bytree': [0.6],
    'regressor__reg_alpha': [0, 0.1, 0.2],  # L1 regularization
    'regressor__reg_lambda': [0.4, 0.5, 0.6]  # L2 regularization
}

# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=10, scoring='r2', verbose=1, n_jobs=-1)
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
plot_path = os.path.join(output_dir, "xgboost_predicted_vs_actual_after_dropping_features.png")
plt.savefig(plot_path)
plt.show()
