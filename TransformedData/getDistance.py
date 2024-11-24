import requests
import re
import pandas as pd

# Define a function to mormalize an address
def normalize_address(address):
    """
    Normalizes an address by removing the extended ZIP code (e.g., "-8118").
    
    Parameters:
    - address (str): The address to normalize.
    
    Returns:
    - str: The normalized address.
    """
    if not isinstance(address, str):
        return address
    # Remove anything after a dash in the ZIP code
    return re.sub(r'\-\d{4}$', '', address).strip()

# Define a function to make the API call
def get_distance(origin, destination):
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origin,
        "destinations": destination,
        "key": 'AIzaSyCJ94_tgx4wIR-zIsEnHi2WKtDCOQeSRp4',
        "units": "imperial"  # Ensure the API returns distances in miles
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "OK" and data["rows"][0]["elements"][0]["status"] == "OK":
                distance_text = data["rows"][0]["elements"][0]["distance"]["text"]
                # Extract numeric part and ensure it's in miles
                if "mi" in distance_text:
                    return float(distance_text.replace(" mi", "").strip().replace(',', ''))
                else:
                    return float(0)
            else:
                print(f"Error in response: {data['rows'][0]['elements'][0]['status']}")
                return None
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"An error occurred while making the API call: {e}")
        return None

# Define a function to add 'Distance' feature
def add_Distance_feature(file_path, columns_to_keep, output_file_path):

    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # Filter the dataset to keep only the specified columns
        df = df[columns_to_keep]

        # Assume 'Property Address' and 'Owner Address' columns exist in the dataset
        if 'Property Address' in df.columns and 'Owner Address' in df.columns:
            df['Distance'] = df.apply(
                lambda row: 0 if normalize_address(row['Property Address']) == normalize_address(row['Owner Address']) 
                else get_distance(row['Property Address'], row['Owner Address']),
                axis=1
            )
        else:
            print("Error: The dataset must contain 'Property Address' and 'Owner Address' columns for the API call.")
            return

        # Save the transformed dataset
        df.to_csv(output_file_path, index=False)
        print(f"Transformed dataset saved to {output_file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # List of input files
    file_list = ["LatapultData/latapult_results (2).csv"]
    output_file_path = ["TransformedData/distance_latapult_results (2).csv"]
    columns_to_keep = ['Property Address', 'Owner Address']

    # Loop through each file in the list and process
    for i, file_path in enumerate(file_list):
        add_Distance_feature(file_path=file_path, columns_to_keep=columns_to_keep, output_file_path=output_file_path[i])