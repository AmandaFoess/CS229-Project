import pandas as pd
import os  # To handle file deletion

# File paths
file_path = '(2) WebScraping/MeckArcGISSales.csv'  # Input file path
intermediate_file = '(2) WebScraping/Intermediate_MeckArcGISSales.csv'  # Temporary intermediate file
final_output_file = '(2) WebScraping/Mod_MeckArcGISSales_Sorted.csv'  # Final output file
transferid_file = '(2) WebScraping/TransferID_MeckArcGISSales.csv'  # Output file for transferid count > 1

# Process the dataset in chunks
chunksize = 10_000  # Adjust chunk size as needed
with pd.read_csv(file_path, chunksize=chunksize, low_memory=False) as reader:
    for i, chunk in enumerate(reader):
        print(f"Processing chunk {i}...")

        # Drop rows with missing 'transferid' or 'parcelid'
        chunk = chunk.dropna(subset=['transferid', 'parcelid'])

        # Remove rows with saleprice == 0
        chunk = chunk[chunk['saleprice'] != 0]

        # Add a new feature 'Acres' calculated from 'shape_Area'
        chunk['Acres'] = (chunk['shape_Area'] / 43560).round(3)  # Convert sq ft to acres

        # Add a new feature 'VDL Sale Price' and set it to blank
        chunk['VDL Sale Price'] = None

        # Calculate the VDL Sale Price for each unique transferid in the chunk
        transferid_counts = chunk['transferid'].value_counts()
        for transferid, count in transferid_counts.items():
            mask = chunk['transferid'] == transferid
            chunk.loc[mask, 'VDL Sale Price'] = chunk.loc[mask, 'saleprice'] / count

        # Write the processed chunk to the intermediate file
        if i == 0:
            # Write the header for the first chunk
            chunk.to_csv(intermediate_file, index=False, mode='w')
        else:
            # Append subsequent chunks without headers
            chunk.to_csv(intermediate_file, index=False, mode='a', header=False)

        print(f"Chunk {i} processed and written to {intermediate_file}")

# Step 2: Read the intermediate file back for sorting
print(f"Sorting all records in the intermediate file...")
sorted_data = pd.read_csv(intermediate_file)
sorted_data['saledate'] = pd.to_datetime(
    sorted_data['saledate'], format='%m/%d/%y', errors='coerce'
) # Ensure saledate is in datetime format
sorted_data = sorted_data.sort_values(by=['saledate', 'transferid'])

# Step 3: Write the sorted data to the final output file
sorted_data.to_csv(final_output_file, index=False)
print(f"Final sorted data saved to {final_output_file}")

# Step 4: Filter records where transferid count > 1
print(f"Filtering records where transferid count > 1...")
transferid_counts = sorted_data['transferid'].value_counts()
transferid_greater_than_1 = sorted_data[sorted_data['transferid'].isin(
    transferid_counts[transferid_counts > 1].index
)]
columns_to_drop = ['propertyid', 'saleprice', 'soldasvaca', 
                   'landuseful', 'landuse', 'deeddescri', 'legalrefer', 
                   'salesvalid', 'shape_Leng', 'shape_Area']
transferid_greater_than_1 = transferid_greater_than_1.drop(columns=columns_to_drop, errors='ignore')
transferid_greater_than_1.to_csv(transferid_file, index=False)
print(f"Records with transferid count > 1 saved to {transferid_file}")

# Step 5: Delete the intermediate file
if os.path.exists(intermediate_file):
    os.remove(intermediate_file)
    print(f"Intermediate file {intermediate_file} deleted.")
else:
    print(f"Intermediate file {intermediate_file} not found, skipping deletion.")
