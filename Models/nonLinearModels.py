import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Load the dataset
input_csv = "(3) TransformedData/Addresses_with_Ratio.csv"  # Update this path if needed
df = pd.read_csv(input_csv)

# Define the target variable and features
target = "House to Lot Ratio"
features = ["Acres", "VDL Sale Price", "Finished Home Value", "Adjusted Finished Home Value", "Latitude", "Longitude"]

# Ensure no missing values in selected features
df = df.dropna(subset=features + [target])

# Separate features and target variable
X = df[features]
y = df[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters for XGBoost
params = {
    "objective": "reg:squarederror",  # Regression with squared error
    "eval_metric": "rmse",           # Evaluation metric
    "learning_rate": 0.05,           # Learning rate
    "max_depth": 6,                  # Maximum depth of a tree
    "subsample": 0.8,                # Subsample ratio of training data
    "colsample_bytree": 0.8,         # Subsample ratio of columns
    "seed": 42                       # Random seed for reproducibility
}

# Train the model with early stopping
print("Training the XGBoost model...")
evals = [(dtrain, "train"), (dtest, "eval")]
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,           # Maximum number of boosting rounds
    early_stopping_rounds=50,      # Stop if no improvement in 50 rounds
    evals=evals,
    verbose_eval=10                # Print progress every 10 rounds
)

# Make predictions
y_pred = model.predict(dtest)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# Feature Importance
print("\nFeature Importance:")
importance = model.get_score(importance_type="weight")
importance_df = pd.DataFrame({
    "Feature": importance.keys(),
    "Importance": importance.values()
}).sort_values(by="Importance", ascending=False)

print(importance_df)

# Save the model
model_file = "(3) TransformedData/xgb_model.json"
model.save_model(model_file)
print(f"Model saved to {model_file}")
