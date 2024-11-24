import requests
from bs4 import BeautifulSoup
import pandas as pd
import argparse

def fetch_all_parcel_data(parcel_number):
    # URL with dynamic ParcelNum query
    url = f"https://taxbill.co.mecklenburg.nc.us/publicwebaccess/BillSearchResults.aspx?ParcelNum={parcel_number}"
    
    # Simulate a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Send GET request
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Locate the parent table with ID `tblSearchResults`
        parent_table = soup.find("table", {"id": "tblSearchResults"})
        if not parent_table:
            print(f"No table found for Parcel Number: {parcel_number}")
            return None
        
        # Locate the nested table with ID `G_dgResults` inside the parent table
        nested_table = parent_table.find("table", {"id": "G_dgResults"})
        if not nested_table:
            print(f"No nested table found for Parcel Number: {parcel_number}")
            return None
        
        # Extract rows from the nested table
        rows = nested_table.find_all("tr")
        if not rows:
            print(f"No rows found in the nested table for Parcel Number: {parcel_number}")
            return None
        
        # Extract headers from the first row
        headers = ["Bill Number", "Old Bill Number", "Parcel Number", "Name", "Location", "Bill Flags", "Current Due"]
        
        # Extract data from the remaining rows
        data = []
        for row in rows[1:]:
            cells = row.find_all("td")
            data_row = [cell.text.strip() for cell in cells]
            if data_row:  # Skip empty rows
                data.append(data_row)
        
        # Debugging information
        print("Headers:", headers)
        print("Number of headers:", len(headers))
        print("First data row:", data[0] if data else "No data rows")
        print("Number of columns in first data row:", len(data[0]) if data else 0)
        
        # Fix header and data mismatch dynamically
        if data and len(headers) != len(data[0]):
            min_columns = min(len(headers), len(data[0]))
            headers = headers[:min_columns]
            data = [row[:min_columns] for row in data]
        
        # Create a DataFrame
        if headers and data:
            df = pd.DataFrame(data, columns=headers)
            return df
        else:
            print(f"No valid data for Parcel Number: {parcel_number}")
            return None
    else:
        print(f"Failed to fetch data for Parcel Number: {parcel_number}. Status code: {response.status_code}")
        return None

def fetch_parcel_ownership_duration(parcel_number):
    # URL with dynamic ParcelNum query
    url = f"https://taxbill.co.mecklenburg.nc.us/publicwebaccess/BillSearchResults.aspx?ParcelNum={parcel_number}"
    
    # Simulate a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Send GET request
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Locate the parent table with ID `tblSearchResults`
        parent_table = soup.find("table", {"id": "tblSearchResults"})
        if not parent_table:
            print(f"No table found for Parcel Number: {parcel_number}")
            return None
        
        # Locate the nested table with ID `G_dgResults` inside the parent table
        nested_table = parent_table.find("table", {"id": "G_dgResults"})
        if not nested_table:
            print(f"No nested table found for Parcel Number: {parcel_number}")
            return None
        
        # Extract rows from the nested table
        rows = nested_table.find_all("tr")
        if not rows:
            print(f"No rows found in the nested table for Parcel Number: {parcel_number}")
            return None
        
        # Extract data from all rows (assume no headers available in <th>)
        data = []
        for row in rows:
            cells = row.find_all("td")
            data_row = [cell.text.strip() for cell in cells]
            if data_row:  # Skip empty rows
                data.append(data_row)
        
        # Define headers manually
        headers = ["Bill Number", "Old Bill Number", "Parcel Number", "Name", "Location", "Bill Flags", "Current Due"]

        # Create a DataFrame
        if headers and data:
            df = pd.DataFrame(data, columns=headers)
            
            # Filter rows where "Bill Flags" is "OWNERSHIP TRANSFER"
            ownership_transfers = df[df["Bill Flags"] == "OWNERSHIP TRANSFER"].copy()
            
            if ownership_transfers.empty:
                print(f"No 'OWNERSHIP TRANSFER' records for Parcel Number: {parcel_number}")
                return None
            
            # Extract year from "Bill Number" and sort transfers by year
            ownership_transfers["Year"] = ownership_transfers["Bill Number"].apply(lambda x: int(x.split("-")[1]))
            ownership_transfers.sort_values(by="Year", ascending=False, inplace=True)

            # Find the two most recent ownership transfers
            if len(ownership_transfers) < 2:
                print(f"Not enough ownership transfers to calculate duration for Parcel Number: {parcel_number}")
                return None
            
            # Calculate the duration between the most recent and the second most recent transfers
            recent_transfer = ownership_transfers.iloc[0]
            previous_transfer = ownership_transfers.iloc[1]
            
            recent_year = recent_transfer["Year"]
            previous_year = previous_transfer["Year"]
            
            # Calculate the duration
            ownership_duration = recent_year - previous_year
            
            print(f"The previous owner held the property for {ownership_duration} years before selling it.")
            return ownership_duration
        else:
            print(f"No valid data for Parcel Number: {parcel_number}")
            return None
    else:
        print(f"Failed to fetch data for Parcel Number: {parcel_number}. Status code: {response.status_code}")
        return None


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run specific methods in the script.")
    
    # Add arguments for the method and any other options
    parser.add_argument(
        "method", 
        choices=["fetch_all_parcel_data", "fetch_parcel_ownership_duration"], 
        help="The method to run: 'fetch_all_parcel_data' for parcel data analysis, 'fetch_parcel_ownership_duration' for a different method."
    )
    parser.add_argument(
        "--parcel_number", 
        type=str, 
        required=False, 
        help="The parcel number to analyze (required for 'main')."
    )
    
    args = parser.parse_args()
    
    # Example usage with multiple parcel numbers
    if args.parcel_number:
        parcel_numbers = [args.parcel_number]
    else:
        # Read parcel numbers from CSV and ensure they are strings
        parcel_numbers = pd.read_csv(
            "LatapultData/pn_latapult_results (2).csv",
            dtype={"Parcel Number": str}  # Ensure Parcel Numbers are read as strings
        )["Parcel Number"].dropna().str.replace("-", "").unique().tolist()


    ownership_durations = []  # For scalar results

    # Call the selected method
    if args.method == "fetch_all_parcel_data":
        all_data = []  # For DataFrames
        # Collect data for all parcels
        for parcel in parcel_numbers:
            parcel_df = fetch_all_parcel_data(parcel)
            if parcel_df is not None:
                parcel_df["Parcel Number"] = parcel
                all_data.append(parcel_df)
        df = pd.concat(all_data, ignore_index=True)

    elif args.method == "fetch_parcel_ownership_duration":
        ownership_durations = []  # For scalar results
        # Collect data for all parcels
        for parcel in parcel_numbers:
            parcel_df = fetch_parcel_ownership_duration(parcel)
            if parcel_df is not None:
                ownership_durations.append({"Parcel Number": parcel, "Ownership Duration": parcel_df})
        df = pd.DataFrame(ownership_durations)

    # Combine all data into a single DataFrame
    print(df)
    df.to_csv("LatapultData/WebScraping/mecklenburg_parcel_data.csv", index=False)