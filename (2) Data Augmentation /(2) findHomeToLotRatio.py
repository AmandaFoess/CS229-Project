import pandas as pd

# File paths
input_file = '(3) TransformedData/TransferID_with_Home_Value.csv'  # Input file with Finished Home Value
output_file = '(3) TransformedData/TransferID_with_Ratio.csv'  # Output file

# Step 1: Load the input file
data = pd.read_csv(input_file)

# Step 2: Remove records where 'Home Transfer ID' is blank
data = data.dropna(subset=['Home Transfer ID'])

# Ensure 'Home Transfer ID' is rounded and treated as a string
data['Home Transfer ID'] = data['Home Transfer ID'].round(0).astype(int).astype(str)

# Step 3: Calculate the count of each unique 'Home Transfer ID'
home_transfer_counts = data['Home Transfer ID'].value_counts()

# Step 4: Divide 'Finished Home Value' by the count for each 'Home Transfer ID'
data['Adjusted Finished Home Value'] = data.apply(
    lambda row: row['Finished Home Value'] / home_transfer_counts[row['Home Transfer ID']],
    axis=1
)

# Step 5: Calculate the House to Lot Ratio
# Avoid division by zero or null values
valid_rows = data['VDL Sale Price'] > 0  # Only calculate for non-zero VDL Sale Price
data['House to Lot Ratio'] = None
data.loc[valid_rows, 'House to Lot Ratio'] = (
    data.loc[valid_rows, 'Adjusted Finished Home Value'] / data.loc[valid_rows, 'VDL Sale Price']
).round(3)  # Round the ratio to 3 decimal places

# Step 6: Save the modified data to a new file
data.to_csv(output_file, index=False)
print(f"Modified data with Adjusted Finished Home Value and House to Lot Ratio saved to {output_file}.")
