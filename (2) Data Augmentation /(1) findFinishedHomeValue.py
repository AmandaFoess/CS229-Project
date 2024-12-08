import pandas as pd

# File paths
transferid_file = '(2) WebScraping/TransferID_MeckArcGISSales.csv'  # Input TransferID file
mod_sales_file = '(2) WebScraping/Mod_MeckArcGISSales_Sorted.csv'  # Input Mod_meckarcgissales file
output_file = '(3) TransformedData/TransferID_with_Home_Value.csv'  # Output file

# Chunk size
chunksize = 1_000

# Step 1: Load the full Mod_MeckArcGISSales data into memory
mod_sales_data = pd.read_csv(mod_sales_file)
mod_sales_data['saledate'] = pd.to_datetime(mod_sales_data['saledate'])  # Ensure datetime format

# Step 2: Process the TransferID data in chunks
with pd.read_csv(transferid_file, chunksize=chunksize) as reader:
    for i, chunk in enumerate(reader):
        print(f"Processing chunk {i + 1}...")

        # Ensure 'saledate' is in datetime format
        chunk['saledate'] = pd.to_datetime(chunk['saledate'])

        # Add new feature columns for Finished Home Value and Home Transfer ID
        chunk['Finished Home Value'] = None
        chunk['Home Transfer ID'] = None

        # Step 3: Process each row in the chunk
        for j, row in chunk.iterrows():
            parcel_id = row['parcelid']
            transfer_saledate = row['saledate']
            
            # Find matching records in Mod_MeckArcGISSales with the same parcelid
            matching_records = mod_sales_data[mod_sales_data['parcelid'] == parcel_id]
            
            if not matching_records.empty:
                # Find records with saledate > transfer_saledate
                valid_records = matching_records[matching_records['saledate'] > transfer_saledate]
                
                if not valid_records.empty:
                    # Find the record with the earliest saledate
                    earliest_record = valid_records.loc[valid_records['saledate'].idxmin()]
                    chunk.at[j, 'Finished Home Value'] = earliest_record['saleprice']
                    chunk.at[j, 'Home Transfer ID'] = earliest_record['transferid']
                else:
                    # No valid records with saledate > transfer_saledate
                    chunk.at[j, 'Finished Home Value'] = None
                    chunk.at[j, 'Home Transfer ID'] = None
            else:
                # No matching records found
                chunk.at[j, 'Finished Home Value'] = None
                chunk.at[j, 'Home Transfer ID'] = None

        # Step 4: Append the processed chunk to the output file
        if i == 0:
            # Write the header for the first chunk
            chunk.to_csv(output_file, index=False, mode='w')
        else:
            # Append subsequent chunks without headers
            chunk.to_csv(output_file, index=False, mode='a', header=False)

        print(f"Chunk {i + 1} processed and saved.")

print(f"Processing complete. Final output saved to {output_file}.")
