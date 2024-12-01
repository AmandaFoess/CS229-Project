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

# Step 1: Identify features to drop based on feature importance
# List of features deemed unimportant based on the feature importance plot
landuse_dict = {
    "MEDICAL CONDOMINIUM": "drop",
    "USE VALUE HOMESITE": "keep",
    "CONDOMINIUM": "drop",
    "RURAL HOMESITE": "keep",
    "CHURCH": "drop",
    "SINGLE FAMILY RESIDENTIAL - ACREAGE": "keep",
    "COMMERCIAL": "drop",
    "TOWN HOUSE SFR": "keep",
    "INDUSTRIAL": "drop",
    "AGRICULTURAL - COMMERCIAL PRODUCTION": "drop",
    "GOLF COURSE CLASS 2 - PRIVATE CLUB": "drop",
    "MULTI FAMILY": "keep",
    "SINGLE FAMILY RESIDENTIAL - GOLF": "keep",
    "WAREHOUSING": "drop",
    "OFFICE CONDOMINIUM": "drop",
    "SINGLE FAMILY RESIDENTIAL": "keep",
    "FLUM/SWIM FLOODWAY (NO BUILD ZONE)": "drop",
    "LABORATORY / RESEARCH": "drop",
    "FOREST - COMMERCIAL PRODUCTION": "drop",
    "SINGLE FAMILY RESIDENTIAL - COMMON": "keep",
    "RESERVED PARCEL": "drop",
    "MULTI FAMILY GARDEN": "keep",
    "MOBILE HOME SUBDIVISION": "drop",
    "OFFICE": "drop",
    "WASTELAND, SLIVERS, GULLIES, ROCK OUTCROP": "drop",
    "NEW PARCEL": "drop",
    "100 YEAR FLOOD PLAIN - AC": "drop",
    "GOLF COURSE CLASS 1 - CHAMPIONSHIP": "drop",
    "ROADWAY CORRIDOR": "drop",
    "WAREHOUSE CONDOMINIUM": "drop",
    "SINGLE FAMILY RESIDENTIAL - WATERFRONT": "keep",
    "NO LAND INTEREST": "drop",
    "MULTI FAMILY DUPLEX/TRIPLEX": "keep",
    "MINI WAREHOUSE": "drop",
    "LUMBER YARD": "drop",
    "COMMERCIAL CONDOMINIUM": "drop",
    "MULTI FAMILY MARINA LAND": "keep",
    "AIR RIGHTS PARCEL": "drop",
    "SUBMERGED LAND, RIVERS AND LAKES": "drop",
    "UNSUITABLE FOR SEPTIC": "drop",
    "SCHOOL, COLLEGE, PRIVATE": "keep",
    "BUFFER STRIP": "drop",
    "SINGLE FAMILY RESIDENTIAL - RIVER": "keep"
}

# Step 2: Create a list of landuse features to drop
features_to_drop = [key for key, value in landuse_dict.items() if value == "drop"]

# Step 3: Filter the DataFrame to exclude records with "drop" landuse features
df = df[~df['landuseful'].isin(features_to_drop)]

# Filter the DataFrame based on the dictionary
# df = df[df['landuseful'].map(landuse_dict).eq("keep")]

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
    'regressor__n_estimators': [100, 300, 500],
    'regressor__max_depth': [4, 6, 8],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__subsample': [0.6, 0.8, 1.0],
    'regressor__colsample_bytree': [0.6, 0.8, 1.0],
    'regressor__reg_alpha': [0, 0.1, 1],  # L1 regularization
    'regressor__reg_lambda': [1, 10, 100]  # L2 regularization
}

# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', verbose=1, n_jobs=-1)
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
