import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def linear_regression_model(X_train, y_train, X_test):
    # Train Linear Regression Model
    model = Ridge(alpha=1.0)  # L2-regularized linear regression (Ridge)
    model.fit(X_train, y_train)

    # Make Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return y_train_pred, y_test_pred

def random_forest_model(X_train, y_train, X_test):
    # Train Random Forest Regressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Make Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return y_train_pred, y_test_pred

# Load the dataset
file_path = "(3) TransformedData/OwnershipLatapultMerged.csv"
df = pd.read_csv(file_path)

# Define columns to exclude
drop_columns = ["Parcel Number", "Owner Name", "County", "City"]
df = df.drop(columns=drop_columns, errors="ignore")

# Define the target column
target = 'Ownership Duration'

# Identify categorical columns
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

# Apply one-hot encoding to categorical columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Get available columns from the dataset
available_columns = df.columns.tolist()

# Dynamically select features
features = [col for col in df.columns.tolist() if col != target]

# Debugging outputs
print(f"Available columns: {available_columns}")
print(f"Selected features: {features}")

# Identify columns for log transformation
log_transform_columns = ['Sale Price', 'Sale Price Per Acre', '1st Mortgage Amount']

# Filter out rows with missing values in the selected features or target
df = df[features + [target]].dropna()

# Split the data into features (X) and target (y)
X = df[features]
y = df[target]

# Standardize Features
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)
X_scaled = X

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Choose and train a model
# Uncomment the desired model
# y_train_pred, y_test_pred = linear_regression_model(X_train=X_train, y_train=y_train, X_test=X_test)
y_train_pred, y_test_pred = random_forest_model(X_train=X_train, y_train=y_train, X_test=X_test)

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
plt.scatter(y_test, y_test_pred, alpha=0.6)
# Plot the identity line (y = x)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label="Ideal Fit (y = x)")
plt.title('Predicted vs Actual Ownership Duration')
plt.xlabel('Actual Ownership Duration')
plt.ylabel('Predicted Ownership Duration')
plt.legend()
plt.tight_layout()

# Save the plot
plot_path = f"Models/Outputs/predicted_vs_actual.png"
plt.savefig(plot_path)
plt.show()
