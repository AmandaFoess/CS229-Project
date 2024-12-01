import pandas as pd
import os

# Load the dataset
input_csv = '(3) TransformedData/Addresses_with_Ratios_and_New_Features.csv'
df = pd.read_csv(input_csv)

# Extract unique values from the 'grantee' column
unique_grantees = df['grantee'].dropna().unique()

# Display the unique values
print(f"Number of unique grantee values: {len(unique_grantees)}")
print("Unique grantee values extracted.")

# Save the unique values to a text file
output_dir = "(3) TransformedData"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "unique_grantee_values.txt")

with open(output_file, "w") as f:
    f.write(f"Number of unique grantee values: {len(unique_grantees)}\n\n")
    f.write("Unique grantee values:\n")
    for grantee in unique_grantees:
        f.write(f"{grantee}\n")

print(f"Unique grantee values saved to: {output_file}")
