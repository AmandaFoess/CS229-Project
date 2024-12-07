import pandas as pd

# File paths
input_file = "(3) TransformedData/Addresses_with_Mapped_Grantees.csv"  # input file
output_file = "(3) TransformedData/Addresses_with_Mapped_Grantees_Cleaned.csv"

# Load the addresses dataset
try:
    df = pd.read_csv(input_file)
    print("Loaded addresses dataset.")
except FileNotFoundError:
    print(f"File {input_file} not found. Ensure the file exists and try again.")
    exit()

# Drop the column named "Home Transfer ID"
df.drop('Home Transfer ID', axis=1, inplace=True)

# Drop the column named "Finished Home Value"
df.drop('Finished Home Value', axis=1, inplace=True)

# Drop the column named "Address"
df.drop('Address', axis=1, inplace=True)

# Handle missing values
df.dropna(subset=['Acres', 'VDL Sale Price', 'Latitude', 'Longitude', 'landuseful', 'Mapped_Grantee', 'naldesc', 'Adjusted Finished Home Value'], inplace=True)

# Transform the 'saledate' column to year-month format (YYYY-MM)
if "saledate" in df.columns:
    df['saledate'] = pd.to_datetime(df['saledate'])  # Ensure it is in datetime format
    df['saledate'] = df['saledate'].dt.to_period('M').astype(str)  # Extract year and month
    print("Transformed 'saledate' to year-month format.")

# Handle outliers for the target feature (House to Lot Ratio)
def remove_outliers_using_iqr(df, column, b, t):
    """Removes outliers in the top and bottom 5% based on IQR."""
    Q1 = df[column].quantile(b)  # Bottom 5% quantile
    Q3 = df[column].quantile(1 - t)  # Top 5% quantile
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers in the target feature
target_feature = "House to Lot Ratio"
if target_feature in df.columns:
    original_size = len(df)
    df = remove_outliers_using_iqr(df, target_feature, 0.1, 0.3)
    removed_records = original_size - len(df)
    print(f"Outlier removal: {removed_records} records removed from '{target_feature}'.")

if "Acres" in df.columns:
    # original_size = len(df)
    # df = df[df["Acres"] <= 1]
    # removed_records = original_size - len(df)
    # Drop all records with Acres > 1
    df = remove_outliers_using_iqr(df, 'Acres', 0.05, 0.1)
    print(f"Records with 'Acres' outside of IQR were removed: {removed_records}")

if "VDL Sale Price" in df.columns:
    # original_size = len(df)
    # df = df[df["VDL Sale Price"] <= 1_000_000]
    # removed_records = original_size - len(df)
    df = remove_outliers_using_iqr(df, 'VDL Sale Price', 0.05, 0.1)
    print(f"Records with 'VDL Sale Price' outside of IQR were removed: {removed_records}")

if "Longitude" in df.columns:
    # original_size = len(df)
    # df = df[df["VDL Sale Price"] <= 1_000_000]
    # removed_records = original_size - len(df)
    df = remove_outliers_using_iqr(df, 'Longitude', 0.05, 0.05)
    print(f"Records with 'Longitude' outside of IQR were removed: {removed_records}")

if "Latitude" in df.columns:
    # original_size = len(df)
    # df = df[df["VDL Sale Price"] <= 1_000_000]
    # removed_records = original_size - len(df)
    df = remove_outliers_using_iqr(df, 'Latitude', 0.05, 0.05)
    print(f"Records with 'Latitude' outside of IQR were removed: {removed_records}")

# Save the updated dataset
df.to_csv(output_file, index=False)
print(f"Updated dataset saved to {output_file}.")
