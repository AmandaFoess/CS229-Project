import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('(3) TransformedData/Addresses_with_Mapped_Grantees_Cleaned.csv')

# Drop the 'VDL Sale Price' column from the DataFrame
if 'VDL Sale Price' in df.columns:
    df.drop('VDL Sale Price', axis=1, inplace=True)
    print("Column 'VDL Sale Price' has been dropped.")


# Handle outliers for the target feature (House to Lot Ratio)
def remove_outliers_using_iqr(df, column):
    """Removes outliers in the top and bottom 5% based on IQR."""
    Q1 = df[column].quantile(0.05)  # Bottom 5% quantile
    Q3 = df[column].quantile(0.95)  # Top 5% quantile
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers in the target feature
target_feature = "House to Lot Ratio"
if target_feature in df.columns:
    original_size = len(df)
    df = remove_outliers_using_iqr(df, target_feature)
    removed_records = original_size - len(df)
    print(f"Outlier removal: {removed_records} records removed from '{target_feature}'.")

# Summary statistics
print("Summary Statistics:")
print(df.describe(include='all'))

# Missing value analysis
missing_values = df.isnull().sum().reset_index()
missing_values.columns = ['Feature', 'Missing Values']
missing_values['Missing Percent'] = (missing_values['Missing Values'] / len(df)) * 100
missing_values = missing_values.sort_values(by='Missing Percent', ascending=False)

print("\nMissing Values Analysis:")
print(missing_values)

# Save missing values analysis to a CSV file
missing_values.to_csv("missing_values_analysis.csv", index=False)

# Extract numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=[np.number])

# Check if numeric_df is empty
if numeric_df.empty:
    print("\nNo numeric columns available for correlation analysis.")
else:
    # Compute the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Print correlation matrix
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Plot the correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.show()

# Distribution of key numeric features
numeric_features = ['Acres', 'VDL Sale Price', 'Finished Home Value', 'House to Lot Ratio']
for feature in numeric_features:
    if feature in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[feature].dropna(), kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f"{feature}_distribution.png")
        plt.show()
    else:
        print(f"Feature {feature} is not in the DataFrame.")

# Distribution of categorical features
categorical_features = ['landuseful', 'Mapped_Grantee']
for feature in categorical_features:
    if feature in df.columns:
        plt.figure(figsize=(10, 6))
        df[feature].value_counts().head(20).plot(kind='bar', color='green')
        plt.title(f'Distribution of {feature} (Top 20)')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f"{feature}_distribution.png")
        plt.show()
    else:
        print(f"Feature {feature} is not in the DataFrame.")

# Save distribution insights to CSV files
for feature in numeric_features:
    if feature in df.columns:
        df[feature].describe().to_csv(f"{feature}_summary.csv")

for feature in categorical_features:
    if feature in df.columns:
        df[feature].value_counts().to_csv(f"{feature}_value_counts.csv")
