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

# Drop unnecessary columns
df.drop(['Home Transfer ID', 'Finished Home Value', 'Address'], axis=1, inplace=True)

# Handle missing values
df.dropna(subset=['Acres', 'VDL Sale Price', 'Latitude', 'Longitude', 'landuseful', 'Mapped_Grantee', 'naldesc', 'Adjusted Finished Home Value'], inplace=True)

# Transform the 'saledate' column to year-month format (YYYY-MM)
if "saledate" in df.columns:
    df['saledate'] = pd.to_datetime(df['saledate'])  # Ensure it is in datetime format
    df['saledate'] = df['saledate'].dt.to_period('M').astype(str)  # Extract year and month
    print("Transformed 'saledate' to year-month format.")

# Function to handle outliers using IQR
def remove_outliers_using_iqr(df, column, b, t):
    """Removes outliers based on IQR ranges for the bottom and top."""
    Q1 = df[column].quantile(b)
    Q3 = df[column].quantile(1 - t)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply the best IQR values
# Best IQR ranges for each variable based on prior analysis
""" iqr_config = {
    "House to Lot Ratio": (0.20, 0.25),
    "Acres": (0.20, 0.10),
    "VDL Sale Price": (0.20, 0.01),
    "Longitude": (0.05, 0.05),
    "Latitude": (0.05, 0.05),
} """

iqr_config = {
    "House to Lot Ratio": (0.20, 0.20),
    "Acres": (0.20, 0.15),
    "VDL Sale Price": (0.10, 0.15),
    "Longitude": (0.05, 0.05),
    "Latitude": (0.05, 0.05),
}

for column, (b, t) in iqr_config.items():
    if column in df.columns:
        original_size = len(df)
        df = remove_outliers_using_iqr(df, column, b, t)
        removed_records = original_size - len(df)
        print(f"Outlier removal for '{column}': {removed_records} records removed.")

# Save the updated dataset
df.to_csv(output_file, index=False)
print(f"Updated dataset saved to {output_file}.")
