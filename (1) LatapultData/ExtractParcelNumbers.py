# import pandas as pd

# # Path to your CSV file
# csv_file = "LatapultData/latapult_results (2).csv"

# # Read the CSV file
# data = pd.read_csv(csv_file)

# # Extract the "Parcel Number" column
# parcel_numbers = data["Parcel Number"].dropna().astype(str).str.replace("-", "").unique().tolist()  # Remove NaNs and get unique values

# # Display the parcel numbers
# print(f"Extracted {len(parcel_numbers)} parcel numbers:")
# #print(parcel_numbers)

# # Optionally, save the list to a new CSV or text file
# pd.DataFrame({"Parcel Number": parcel_numbers}).to_csv("LatapultData/pn_latapult_results (2).csv", index=False)

import os
import pandas as pd

# Directory containing all CSV files
folder_path = "LatapultData/OgData/"

# List to store all parcel numbers
all_parcel_numbers = []

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is a CSV
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        
        # Read the CSV file
        data = pd.read_csv(file_path)

        # print name of .CSV file for log
        print(f"Extracting {len(data)} Parcel Numbers from '{file_path}'")
        
        # Extract the "Parcel Number" column, process, and add to the list
        if "Parcel Number" in data.columns:
            parcel_numbers = (
                data["Parcel Number"]
                .dropna()
                .astype(str)
                .str.replace("-", "", regex=False)
                .unique()
                .tolist()
            )
            all_parcel_numbers.extend(parcel_numbers)

print(f"Total parcel numbers extracted: {len(all_parcel_numbers)}")

# Get unique parcel numbers from all files
unique_parcel_numbers = list(set(all_parcel_numbers))

# Create a DataFrame for the unique parcel numbers
output_df = pd.DataFrame({"Parcel Number": unique_parcel_numbers})

# Save the DataFrame to a new CSV file
output_csv_path = "LatapultData/ParcelNumbers.csv"
output_df.to_csv(output_csv_path, index=False)

print(f"Extracted {len(unique_parcel_numbers)} unique parcel numbers.")
print(f"Combined parcel numbers saved to: {output_csv_path}")
