import pandas as pd

# File paths
addresses_file = "(3) TransformedData/Addresses_with_Ratios_and_New_Features.csv"  # Input file
standardized_grantee_file = "(3) TransformedData/Addresses_with_Standardized_Grantees.csv"  # Standardized grantee mapping
output_file = "(3) TransformedData/Addresses_with_Mapped_Grantees.csv"  # Output file

# Load the addresses dataset
try:
    addresses_df = pd.read_csv(addresses_file)
    print("Loaded addresses dataset.")
except FileNotFoundError:
    print(f"File {addresses_file} not found. Ensure the file exists and try again.")
    exit()

# Load the standardized grantee mapping
try:
    grantee_mapping_df = pd.read_csv(standardized_grantee_file)
    print("Loaded standardized grantee mapping.")
except FileNotFoundError:
    print(f"File {standardized_grantee_file} not found. Ensure the file exists and try again.")
    exit()

# Validate columns in the mapping file
required_columns = ['Original_Grantee', 'Standardized_Grantee']
if not all(col in grantee_mapping_df.columns for col in required_columns):
    print(f"Error: The standardized grantee file does not contain the required columns: {required_columns}")
    exit()

# Create a mapping dictionary from the grantee mapping file
grantee_mapping = pd.Series(
    grantee_mapping_df['Standardized_Grantee'].values, 
    index=grantee_mapping_df['Original_Grantee']
).to_dict()

# Map the `grantee` column in the addresses DataFrame
addresses_df['Mapped_Grantee'] = addresses_df['grantee'].map(grantee_mapping)

# Save the updated dataset
addresses_df.to_csv(output_file, index=False)
print(f"Updated dataset saved to {output_file}.")
