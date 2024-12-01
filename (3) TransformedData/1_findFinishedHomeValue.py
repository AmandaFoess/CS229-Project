import pandas as pd
import requests
import os

def fetch_unique_addresses(parcel_id):
    """Fetch all unique addresses for a given parcel ID and return them as a list."""
    base_url = "https://gis.charlottenc.gov/arcgis/rest/services/CLT_Ex/CLTEx_MoreInfo/MapServer/4/query"
    params = {
        "where": f"Tax_ID='{parcel_id}'",
        "outFields": "Tax_ID,Location",
        "outSR": "",
        "f": "json"
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if "features" in data and len(data["features"]) > 0:
        # Retrieve addresses and ensure they are unique
        addresses = [feature["attributes"].get("Location", "Unknown Address") for feature in data["features"]]
        return list(set(addresses))  # Remove duplicates by converting to a set, then back to a list
    return []

def get_processed_ids(log_file):
    """Read the log file to get the list of already processed parcel IDs."""
    if os.path.exists(log_file):
        with open(log_file, "r") as file:
            return set(file.read().splitlines())
    return set()

def save_processed_id(parcel_id, log_file):
    """Append a processed parcel ID to the log file."""
    with open(log_file, "a") as file:
        file.write(f"{parcel_id}\n")

# Configuration
transferid_file = '(2) WebScraping/TransferID_MeckArcGISSales.csv'  # Input TransferID file
mod_sales_file = '(2) WebScraping/Mod_MeckArcGISSales_Sorted.csv'  # Input Mod_meckarcgissales file
output_file = '(3) TransformedData/TransferID_with_Home_Value.csv'  # Output file

# Read the input CSV
df = pd.read_csv(input_csv)

# Get the set of already processed IDs
processed_ids = get_processed_ids(log_file)

# Add a new column for the list of unique addresses if not already present
if "Addresses" not in df.columns:
    df["Addresses"] = None

# Process in chunks of 10
chunk_size = 10
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i + chunk_size]

    for index, row in chunk.iterrows():
        parcel_id = str(row["parcelid"])  # Ensure it's a string for consistency
        
        # Skip already processed IDs
        if parcel_id in processed_ids:
            continue

        print(f"Processing Parcel ID: {parcel_id}")

        # Fetch the list of unique addresses
        addresses = fetch_unique_addresses(parcel_id)
        df.at[index, "Addresses"] = addresses  # Save the list in the DataFrame

        # Save the processed ID to the log file
        save_processed_id(parcel_id, log_file)

    # Save the progress to the output CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved progress to {output_csv} (up to index {i + chunk_size})")
