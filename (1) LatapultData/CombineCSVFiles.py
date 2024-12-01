import os
import pandas as pd

def format_parcel_number(parcel_number):
    """
    Format the parcel number to match the required format.

    Args:
        parcel_number (str): The raw parcel number.

    Returns:
        str: The formatted parcel number.
    """
    if pd.isna(parcel_number):
        return None  # Handle missing values gracefully
    parcel_number = str(parcel_number).strip()  # Convert to string and remove leading/trailing spaces
    parcel_number = parcel_number.replace("-", "")  # Remove any dashes
    return parcel_number

def combine_csv_files(input_folder, output_file):
    """
    Combine all CSV files in a folder into a single CSV file and format parcel numbers.

    Args:
        input_folder (str): Path to the folder containing the CSV files.
        output_file (str): Path to save the combined CSV file.
    """
    # List to store DataFrames
    all_dataframes = []

    # Iterate through all files in the folder
    for file_name in os.listdir(input_folder):
        # Check if the file is a CSV
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_folder, file_name)
            print(f"Reading {file_path}")
            
            # Read the CSV file into a DataFrame
            try:
                df = pd.read_csv(file_path)
                
                # Format the Parcel Number column if it exists
                if "Parcel Number" in df.columns:
                    df["Parcel Number"] = df["Parcel Number"].apply(format_parcel_number)

                all_dataframes.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Combine all DataFrames into one
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Combined {len(all_dataframes)} CSV files.")

        # Save the combined DataFrame to a new CSV file
        combined_df.to_csv(output_file, index=False)
        print(f"Combined CSV saved to {output_file}")
    else:
        print("No CSV files found in the folder.")

if __name__ == "__main__":
    # Folder containing the CSV files
    input_folder = "LatapultData/OgData"

    # Output file for the combined CSV
    output_file = "LatapultData/AllLatapultData.csv"

    # Combine the CSV files
    combine_csv_files(input_folder, output_file)
