import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import os

# Load the dataset
df = pd.read_csv('(3) TransformedData/Addresses_with_Mapped_Grantees.csv')

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

# Define features and target variable
features = ['Acres', 'VDL Sale Price', 'Latitude', 'Longitude', 'sale_year', 'landuseful', 'Mapped_Grantee']
X = df[features]
y = df['House to Lot Ratio']

# Preprocess the data
# numeric_features = ['Acres', 'VDL Sale Price', 'Latitude', 'Longitude', 'sale_year']
categorical_features = ['landuseful', 'Mapped_Grantee']

preprocessor = ColumnTransformer(
    transformers=[
        # ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Encode categorical features
    ])

# Create the XGBoost model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, n_estimators=500, max_depth=8, learning_rate=0.1, subsample=1.0, colsample_bytree=0.6))
])

# Train the model
model.fit(X, y) 

# Extract feature names
# numeric_feature_names = numeric_features
numeric_feature_names = []
categorical_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

# Clean categorical feature names by removing the prefix "Mapped_Grantee_"
cleaned_feature_names = [
    name.replace("Mapped_Grantee_", "") for name in categorical_feature_names
]

all_feature_names = numeric_feature_names + cleaned_feature_names

# Extract feature importances
feature_importances = model.named_steps['regressor'].feature_importances_

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Save the feature importance to a CSV file
output_dir = "Models/Outputs"
os.makedirs(output_dir, exist_ok=True)
importance_csv_path = os.path.join(output_dir, "feature_importances.csv")
importance_df.to_csv(importance_csv_path, index=False)

# Plot the feature importance
plt.figure(figsize=(9, 9))
plt.barh(importance_df['Feature'].head(50), importance_df['Importance'].head(50), color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # Flip the order for better readability
plt.tight_layout()

# Save the plot
importance_plot_path = os.path.join(output_dir, "feature_importances.png")
plt.savefig(importance_plot_path)
plt.show()

print(f"Feature importance saved to {importance_csv_path}")
print(f"Feature importance plot saved to {importance_plot_path}")
