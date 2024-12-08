import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from itertools import product
import os

# File paths
input_file = "(3) TransformedData/Addresses_with_Mapped_Grantees.csv"
output_file = "(3) TransformedData/Optimized_Cleaned_Data.csv"
results_csv = "(3) TransformedData/IQR_Grid_Search_Results.csv"  # Results file

# Load the dataset
df = pd.read_csv(input_file)

# Drop unnecessary columns
df.drop(['Home Transfer ID', 'Finished Home Value', 'Address'], axis=1, inplace=True)

# Handle missing values
df.dropna(subset=['Acres', 'VDL Sale Price', 'Latitude', 'Longitude', 'landuseful', 'Mapped_Grantee', 'naldesc', 'Adjusted Finished Home Value'], inplace=True)

# Transform the 'saledate' column to year-month format
df['saledate'] = pd.to_datetime(df['saledate']).dt.to_period('M').astype(str)

# Handle outliers using IQR
def remove_outliers_using_iqr(df, column, bottom_range, top_range):
    """Removes outliers based on feature-specific IQR ranges."""
    Q1 = df[column].quantile(bottom_range)
    Q3 = df[column].quantile(1 - top_range)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Features and target variable
features = ['Acres', 'VDL Sale Price', 'Latitude', 'Longitude', 'saledate_numeric', 'landuseful', 'Mapped_Grantee']
target = 'House to Lot Ratio'

# IQR ranges to test
iqr_ranges = [0.01, 0.05, 0.10, 0.15, 0.20]
results = []
variables = ["House to Lot Ratio", "Acres", "VDL Sale Price"]

# Initialize or load existing results
if os.path.exists(results_csv):
    results_df = pd.read_csv(results_csv)
    completed_configs = set(
        tuple(row) for row in results_df[["House to Lot Ratio (Bottom)", "House to Lot Ratio (Top)", 
                                          "Acres (Bottom)", "Acres (Top)", "VDL Sale Price (Bottom)", 
                                          "VDL Sale Price (Top)"]].to_numpy()
    )
    results = results_df.to_dict('records')
else:
    completed_configs = set()
    results = []

# Iterate over all combinations of IQR ranges for variables
for bottom_ranges in product(iqr_ranges, repeat=len(variables)):
    for top_ranges in product(iqr_ranges, repeat=len(variables)):
        config = tuple(bottom_ranges + top_ranges)
        
        # Skip already completed configurations
        if config in completed_configs:
            continue
        
        temp_df = df.copy()
        
        # Apply outlier removal for each variable
        for i, var in enumerate(variables):
            temp_df = remove_outliers_using_iqr(temp_df, var, bottom_ranges[i], top_ranges[i])
        
        # Skip configurations with too few records
        if temp_df.shape[0] < 100:
            continue

        # Transform 'saledate' into numeric
        temp_df['saledate_numeric'] = (
            (pd.to_datetime(temp_df['saledate']).dt.year - pd.to_datetime(temp_df['saledate']).dt.year.min()) * 12 +
            pd.to_datetime(temp_df['saledate']).dt.month
        )
        temp_df.drop('saledate', axis=1, inplace=True)

        # Prepare data for training
        X = temp_df[features].drop(columns=['saledate'], errors='ignore')
        y = temp_df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocess data
        numeric_features = ['Acres', 'VDL Sale Price', 'Latitude', 'Longitude', 'saledate_numeric']
        categorical_features = ['landuseful', 'Mapped_Grantee']
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        # Model pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(random_state=42, n_estimators=300, max_depth=8, learning_rate=0.1, subsample=1.0, colsample_bytree=0.6))
        ])

        # Fit the pipeline
        model.fit(X_train, y_train)

        # Evaluate model on test set
        y_test_pred = model.predict(X_test)
        r2_test = r2_score(y_test, y_test_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        # Evaluate model on training set
        # y_train_pred = model.predict(X_train)
        # r2_train = r2_score(y_train, y_train_pred)
        # mse_train = mean_squared_error(y_train, y_train_pred)

        # Record results
        results.append({
            "House to Lot Ratio (Bottom)": bottom_ranges[0],
            "House to Lot Ratio (Top)": top_ranges[0],
            "Acres (Bottom)": bottom_ranges[1],
            "Acres (Top)": top_ranges[1],
            "VDL Sale Price (Bottom)": bottom_ranges[2],
            "VDL Sale Price (Top)": top_ranges[2],
            # "Latitude (Bottom)": bottom_ranges[3],
            # "Latitude (Top)": top_ranges[3],
            # "Longitude (Bottom)": bottom_ranges[4],
            # "Longitude (Top)": top_ranges[4],
            "Records Used": temp_df.shape[0],
            # "R^2 Train": r2_train,
            # "MSE Train": mse_train,
            "R^2 Test": r2_test,
            "MSE Test": mse_test
        })
        
        # Append new result to CSV immediately
        pd.DataFrame([results[-1]]).to_csv(results_csv, mode='a', header=not os.path.exists(results_csv), index=False)

        print(f"IQR Results: R^2 Test={r2_test}, Records Used={temp_df.shape[0]}")


# Print the best parameters
best_result = results_df.loc[results_df['R2'].idxmax()]
print("\nBest IQR Parameters:")
print(best_result)
