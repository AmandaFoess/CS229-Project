import pandas as pd
import requests
import os
import googlemaps

def fetch_address(parcel_id):
    """Fetch the first address for a given parcel ID."""
    base_url = "https://gis.charlottenc.gov/arcgis/rest/services/CLT_Ex/CLTEx_MoreInfo/MapServer/4/query"
    params = {
        "where": f"Tax_ID='{parcel_id}'",
        "outFields": "Tax_ID,Location",
        "f": "json"
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if "features" in data and data["features"]:
        return data["features"][0]["attributes"].get("Location", "Unknown Address")
    return "Address Not Found"

def geocode_address(address, gmaps_client):
    """Convert an address to latitude and longitude using Google Geocoding API."""
    geocode_result = gmaps_client.geocode(address)
    if geocode_result:
        location = geocode_result[0]['geometry']['location']
        return location['lat'], location['lng']
    return None, None

# Configuration
input_csv = "(3) TransformedData/TransferID_with_Ratio.csv"
output_csv = "(3) TransformedData/Addresses_with_Ratio.csv"
api_key = "AIzaSyCJ94_tgx4wIR-zIsEnHi2WKtDCOQeSRp4"  # Replace with your actual API key

# Initialize Google Maps client
gmaps_client = googlemaps.Client(key=api_key)

# Read the input CSV
df = pd.read_csv(input_csv)

# Load existing output data if available
if os.path.exists(output_csv):
    df_existing = pd.read_csv(output_csv)
    processed_ids = set(df_existing["parcelid"].astype(str))
else:
    df_existing = pd.DataFrame()
    processed_ids = set()

# Filter out already processed rows
df = df[~df["parcelid"].astype(str).isin(processed_ids)]

# Add new columns for address, latitude, and longitude if not present
if "Address" not in df.columns:
    df["Address"] = None
if "Latitude" not in df.columns:
    df["Latitude"] = None
if "Longitude" not in df.columns:
    df["Longitude"] = None

# Process in chunks of 10
chunk_size = 10
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i + chunk_size]

    for index, row in chunk.iterrows():
        parcel_id = str(row["parcelid"])  # Ensure it's a string for consistency

        print(f"Processing Parcel ID: {parcel_id}")

        # Fetch the first address
        address = fetch_address(parcel_id)
        df.at[index, "Address"] = address

        # Geocode the address to get latitude and longitude
        if address != "Address Not Found":
            latitude, longitude = geocode_address(address, gmaps_client)
            df.at[index, "Latitude"] = latitude
            df.at[index, "Longitude"] = longitude

    # Append processed chunk to output CSV
    if not df_existing.empty or os.path.exists(output_csv):
        chunk.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        chunk.to_csv(output_csv, mode="w", header=True, index=False)

    print(f"Saved chunk to {output_csv} (up to index {i + chunk_size})")

    # Update processed_ids set
    processed_ids.update(chunk["parcelid"].astype(str))

print("Processing complete.")
