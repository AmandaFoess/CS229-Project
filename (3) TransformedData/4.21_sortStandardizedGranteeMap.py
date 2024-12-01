import pandas as pd

# File path for the CSV containing Original and Standardized Grantee Names
grantee_csv = "(3) TransformedData/Addresses_with_Standardized_Grantees.csv"

# Load the dataset
try:
    df = pd.read_csv(grantee_csv)
    print("Loaded existing grantee mapping file.")
except FileNotFoundError:
    print(f"File {grantee_csv} not found. Ensure the file exists and try again.")
    exit()

# Validate columns
required_columns = ['Original_Grantee', 'Standardized_Grantee']
if not all(col in df.columns for col in required_columns):
    print(f"Error: The file does not contain the required columns: {required_columns}")
    exit()

# Iterate over the records and prompt for input
for index, row in df.iterrows():
    original = row['Original_Grantee']
    standardized = row['Standardized_Grantee']
    
    # Skip rows where Standardized_Grantee is already filled
    if isinstance(standardized, str) and standardized.strip():
        continue
    
    print(f"\nRecord {index + 1}:")
    print(f"Original Grantee: {original}")
    print(f"Standardized Grantee: {standardized}")
    
    # Ask the user whether to copy the value
    user_input = input("Do you want to copy the Original Grantee to the Standardized Grantee? (y/n): ").strip().lower()
    
    if user_input == 'n':
        print(f"No change made for: {original}")
    else:
        # Copy the value
        df.at[index, 'Standardized_Grantee'] = original
        print(f"Updated Standardized Grantee for: {original}, {df.at[index, 'Standardized_Grantee']}")
        
        # Save changes to the CSV file after every update
        df.to_csv(grantee_csv, index=False)
        print(f"Saved changes to {grantee_csv}.")
        print(f"No change made for: {original}")

print(f"\nAll updates saved to {grantee_csv}.")
