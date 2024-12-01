import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_features(input_file, label_column, output_summary, output_relevance_plot, output_metrics_plot, output_distribution_plot):
    """
    Analyze statistical summaries, feature relevance, model performance, and zoning code distribution by ownership type.

    Args:
        input_file (str): Path to the merged dataset CSV file.
        label_column (str): The target label column.
        output_summary (str): Path to save the statistical summary file.
        output_relevance_plot (str): Path to save the feature relevance plot.
        output_metrics_plot (str): Path to save the model performance metrics plot.
        output_distribution_plot (str): Path to save the zoning code distribution plot.
    """
    # Load the merged dataset
    print("Loading dataset...")
    df = pd.read_csv(input_file)

    # Handle missing values in the target column
    if df[label_column].isnull().any():
        df = df.dropna(subset=[label_column])

    # Separate features (X) and target (y)
    y = df[label_column]
    X = df.drop(columns=[label_column, "Parcel Number"], errors="ignore")  # Exclude irrelevant columns

    # Convert categorical columns to numeric using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Handle missing values
    X = X.fillna(0)  # Replace NaN with 0

    # Statistical Summaries
    stats = X.describe(include="all").T  # Transpose for better readability
    stats["missing_values"] = X.isnull().sum()  # Count missing values
    stats.to_csv(output_summary)

    # Feature Relevance Analysis
    correlation = pd.Series(np.nan, index=X.columns)
    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.number):
            correlation[col] = np.corrcoef(X[col], y)[0, 1]
    correlation = correlation.sort_values(ascending=False)

    mutual_info = mutual_info_regression(X, y, random_state=42)
    mutual_info_series = pd.Series(mutual_info, index=X.columns).sort_values(ascending=False)

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    relevance_summary = pd.DataFrame({
        "Correlation": correlation,
        "Mutual Information": mutual_info_series,
        "Random Forest Importance": feature_importance
    }).sort_values(by="Random Forest Importance", ascending=False)
    relevance_summary.to_csv(output_summary.replace(".csv", "_relevance.csv"))

    # Plot Top 10 Features by Importance
    plt.figure(figsize=(12, 8))
    relevance_summary["Random Forest Importance"].head(30).plot(kind="barh", title="Top Features by Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_relevance_plot)
    plt.show()

    # Model Performance Metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Plot Model Performance Metrics
    plt.figure(figsize=(8, 6))
    metrics = {"R²": r2, "MAE": mae, "RMSE": rmse}
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    plt.barh(metric_names, metric_values, color=["skyblue", "orange", "green"])
    plt.title("Model Performance Metrics")
    plt.xlabel("Value")
    plt.tight_layout()
    plt.savefig(output_metrics_plot)
    plt.show()

    # Zoning Code Distribution Plot
    if "Property Indicator" in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x="Property Indicator", order=df["Property Indicator"].value_counts().index)
        plt.title("Distribution of Ownership Types by Property Indicator")
        plt.xlabel("Property Indicator")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_distribution_plot)
        plt.show()



if __name__ == "__main__":
    # Input file: merged dataset
    input_file = "(3) TransformedData/OwnershipLatapultMerged.csv"

    # Label column (target)
    label_column = "Ownership Duration"

    # Output files
    output_summary = "(4) StatisticalAnalysis/FeatureAnalysisSummary.csv"
    output_relevance_plot = "(4) StatisticalAnalysis/FeatureRelevancePlot.png"
    output_metrics_plot = "(4) StatisticalAnalysis/ModelPerformanceMetrics.png"
    output_distribution_plot = "(4) StatisticalAnalysis/ZoningDistributionPlot.png"

    # Analyze features
    analyze_features(input_file, label_column, output_summary, output_relevance_plot, output_metrics_plot, output_distribution_plot)
