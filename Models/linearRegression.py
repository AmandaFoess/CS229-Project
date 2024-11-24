import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np

def linear_regression_model(X_train, y_train, X_test):
    # Train Linear Regression Model
    model = LinearRegression()

    # Train Ridge regression with L2 regularization
    model = Ridge(alpha=1.0)

    model.fit(X_train, y_train)

    # Make Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("Linear Regression Model Performance:")

    return y_train_pred, y_test_pred

def random_forest_model(X_train, y_train, X_test):
    # Train Linear Regression Model
    model = RandomForestRegressor()

    # Train Ridge regression with L2 regularization
    model = Ridge(alpha=1.0)

    # Assign weights based on distance bin counts
    distance_bin_weights = y_train.value_counts(normalize=True).to_dict()
    sample_weights = y_train.map(distance_bin_weights)   

    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Make Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("Random Forest Model Performance:")

    return y_train_pred, y_test_pred

# Load the dataset
file_path = "TransformedData/large_test_dataset.csv"
df = pd.read_csv(file_path)

# Select Features and Target
# Features to include in the model (adjust as needed based on your dataset)
features = columns_to_keep = ['Calculated Acres','Sale Price Per Acre']
target = 'Distance'

# Filter out rows with missing values in the selected features or target
df = df[features + [target]].dropna()

# Split the data into features (X) and target (y)
X = df[features]
y = df[target]

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(df[target])
#X_scaled = X

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Make Predictions
#y_train_pred, y_test_pred = linear_regression_model(X_train=X_train, y_train=y_train, X_test=X_test)
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
plt.title('Predicted vs Actual Distance')
plt.xlabel('Actual Distance')
plt.ylabel('Predicted Distance')

# Save the plot
plot_path = f"Models/Outputs/linear_regression_accuracy.png"
plt.savefig(plot_path)
