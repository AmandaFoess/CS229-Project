import pandas as pd
import os

def fetch_parcel_data(parcel_number):
    """
    Simulated data fetch. Replace this with your actual data-fetching logic.
    Returns a dictionary containing the parcel number and dummy data.
    """
    return {
        "Parcel Number": parcel_number,
        "Data": f"Sample data for parcel {parcel_number}"
    }

def save_completed_parcel(parcel_number, completed_file="completed_parcels.csv"):
    """
    Append a completed parcel number to the completed list file.
    """
    with open(completed_file, "a") as f:
        f.write(f"{parcel_number}\n")

def load_completed_parcels(completed_file="completed_parcels.csv"):
    """
    Load the list of completed parcels from the completed list file.
    Returns a set of completed parcel numbers.
    """
    if os.path.exists(completed_file):
        with open(completed_file, "r") as f:
            return set(line.strip() for line in f)
    return set()

def save_parcel_data(parcel_data, output_file="results.csv"):
    """
    Save fetched parcel data to the results file.
    Appends to the file if it already exists; otherwise, creates a new file.
    """
    df = pd.DataFrame([parcel_data])
    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, mode="a", header=False, index=False)

if __name__ == "__main__":
    # Load the full list of parcel numbers from the input CSV
    parcel_numbers = pd.read_csv(
        "LatapultData/pn_latapult_results (2).csv", 
        dtype={"Parcel Number": str}
    )["Parcel Number"].str.replace("-", "").dropna().unique().tolist()

    # Load the list of already completed parcels
    completed_parcels = load_completed_parcels()

    # Determine remaining parcels to process
    remaining_parcels = [p for p in parcel_numbers if p not in completed_parcels]
    print(f"{len(remaining_parcels)} parcels remaining to fetch.")

    # Process each remaining parcel
    for parcel in remaining_parcels:
        try:
            # Fetch data for the current parcel
            parcel_data = fetch_parcel_data(parcel)
            
            # Save the fetched data
            save_parcel_data(parcel_data)
            
            # Mark the parcel as completed
            save_completed_parcel(parcel)
            print(f"Completed parcel {parcel}")
        
        except Exception as e:
            print(f"Error fetching data for parcel {parcel}: {e}")
