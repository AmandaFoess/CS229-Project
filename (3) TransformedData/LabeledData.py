import pandas as pd
from datetime import datetime

def append_latapult_to_ownership(
    ownership_file, 
    latapult_file, 
    output_file, 
    columns_to_remove, 
    ownership_columns_to_remove,
    final_dropped_columns, 
    filter_ownership_type=None, 
    filter_property_indicator=None
):
    """
    Iterate through each ownership record, apply various filters, remove specified columns, 
    and append the filtered Latapult data to the ownership records.

    Args:
        ownership_file (str): Path to the ownership duration CSV file.
        latapult_file (str): Path to the Latapult results CSV file.
        output_file (str): Path to save the final merged CSV file.
        columns_to_remove (list): List of columns to remove from the Latapult data.
        ownership_columns_to_remove (list): List of columns to remove from the ownership data.
        filter_ownership_type (str or None): Ownership type to exclude. If None, no filtering is applied.
        filter_property_indicator (list or None): List of property indicator values to exclude. If None, no filtering is applied.
    """
    # Load ownership duration data
    print("Loading ownership duration data...")
    ownership_df = pd.read_csv(ownership_file)

    # Remove specified columns from ownership data if they exist
    print("Dropping specified columns from ownership data if they exist...")
    ownership_columns_found = [col for col in ownership_columns_to_remove if col in ownership_df.columns]
    ownership_df = ownership_df.drop(columns=ownership_columns_found, errors="ignore")
    if ownership_columns_found:
        print(f"Dropped columns from ownership data: {ownership_columns_found}")
    else:
        print("No specified columns were found to drop in ownership data.")

    # Filter out records with the specified ownership type if provided
    if filter_ownership_type:
        print(f"Filtering out records with ownership type: {filter_ownership_type}")
        before_filtering_count = len(ownership_df)
        ownership_df = ownership_df[ownership_df["Duration Type"] != filter_ownership_type]
        after_filtering_count = len(ownership_df)
        print(f"Filtered out {before_filtering_count - after_filtering_count} ownership records.")

    # Load Latapult results data
    print("Loading Latapult results data...")
    latapult_df = pd.read_csv(latapult_file)

    # Filter out Latapult records with the specified property indicator values if provided
    if filter_property_indicator:
        if "Property Indicator" in latapult_df.columns:
            print(f"Filtering out Latapult records with property indicator values: {filter_property_indicator}")
            before_filtering_count = len(latapult_df)
            latapult_df = latapult_df[~latapult_df["Property Indicator"].isin(filter_property_indicator)]
            after_filtering_count = len(latapult_df)
            print(f"Filtered out {before_filtering_count - after_filtering_count} Latapult records.")
        else:
            print("Warning: 'Property Indicator' column not found in Latapult data. Skipping property indicator filtering.")

    # Remove specified columns from Latapult data if they exist
    print("Dropping specified columns from Latapult data if they exist...")
    columns_found = [col for col in columns_to_remove if col in latapult_df.columns]
    latapult_df = latapult_df.drop(columns=columns_found, errors="ignore")
    if columns_found:
        print(f"Dropped columns from Latapult data: {columns_found}")
    else:
        print("No specified columns were found to drop in Latapult data.")

    # Format Parcel Number in both datasets
    print("Formatting Parcel Numbers...")
    ownership_df["Parcel Number"] = ownership_df["Parcel Number"].astype(str).str.replace("-", "").str.strip()
    latapult_df["Parcel Number"] = latapult_df["Parcel Number"].astype(str).str.replace("-", "").str.strip()

    # Filter ownership records to include only those matching filtered Latapult records
    print("Filtering ownership records based on filtered Latapult data...")
    filtered_parcels = set(latapult_df["Parcel Number"])
    before_filtering_count = len(ownership_df)
    ownership_df = ownership_df[ownership_df["Parcel Number"].isin(filtered_parcels)]
    after_filtering_count = len(ownership_df)
    print(f"Filtered out {before_filtering_count - after_filtering_count} ownership records that did not match filtered Latapult records.")

    # Prepare a list to store combined records
    combined_records = []

    # Iterate through each ownership record
    print("Processing records...")
    for index, ownership_record in ownership_df.iterrows():
        parcel_number = ownership_record["Parcel Number"]

        # Find the matching Latapult record
        matching_latapult_record = latapult_df[latapult_df["Parcel Number"] == parcel_number]

        if not matching_latapult_record.empty:
            # Merge ownership record with the Latapult record
            latapult_data = matching_latapult_record.iloc[0].to_dict()  # Convert Latapult data to dictionary
            combined_record = {**ownership_record.to_dict(), **latapult_data}  # Combine dictionaries
            combined_records.append(combined_record)

    # Create a DataFrame from the combined records
    combined_df = pd.DataFrame(combined_records)

    # Handle null ownership duration values
    # print("Handling null values in 'Ownership Duration'...")
    # current_year = datetime.now().year
    # if "Ownership Duration" in combined_df.columns and "Year Built" in combined_df.columns:
    #     def fill_ownership_duration(row):
    #         if pd.isnull(row["Ownership Duration"]):
    #             if row["Year Built"] and (current_year - row["Year Built"] > 15):
    #                 return current_year - row["Year Built"]
    #             # else:
    #             #     return 15
    #             # return 15
    #         return row["Ownership Duration"]

    #     combined_df["Ownership Duration"] = combined_df.apply(fill_ownership_duration, axis=1)
    #     combined_df = combined_df.dropna(subset=["Ownership Duration"])
    # else:
    #     print("Warning: 'Ownership Duration' or 'Year Built' column not found. Skipping null value handling.")
    
    # Handle null ownership duration values
    print("Removing rows with null values in 'Ownership Duration'...")
    if "Ownership Duration" in combined_df.columns:
        before_removal_count = len(combined_df)
        combined_df = combined_df.dropna(subset=["Ownership Duration"])
        after_removal_count = len(combined_df)
        print(f"Removed {before_removal_count - after_removal_count} rows with null 'Ownership Duration'.")
    else:
        print("Warning: 'Ownership Duration' column not found. Skipping null value handling.")


    combined_df = combined_df.drop(columns=final_dropped_columns, errors="ignore")

    # Handle null mobile home indicator values
    print("Handling null values in 'Mobile Home Indicator'...")
    if "Mobile Home Indicator" in combined_df.columns:
        def fill_mobile_home_indicator(row):
            if (row["Mobile Home Indicator"] == ' '):
                return 0
            return 1

        combined_df["Mobile Home Indicator"] = combined_df.apply(fill_mobile_home_indicator, axis=1)
    else:
        print("Warning: 'Mobile Home Indicator' column not found. Skipping null value handling.")
    
    # Handle null Owner Corporate Indicator values
    print("Handling null values in 'Owner Corporate Indicator'...")
    if "Owner Corporate Indicator" in combined_df.columns:
        def fill_mobile_home_indicator(row):
            if (row["Owner Corporate Indicator"] == ' '):
                return 0
            return 1

        combined_df["Owner Corporate Indicator"] = combined_df.apply(fill_mobile_home_indicator, axis=1)
    else:
        print("Warning: 'Owner Corporate Indicator' column not found. Skipping null value handling.")

    # Handle Out of State Owner values
    print("Handling values in 'Out of State Owner'...")
    if "Out of State Owner" in combined_df.columns:
        def fill_mobile_home_indicator(row):
            if (row["Out of State Owner"] == 'No'):
                return 0
            return 1

        combined_df["Out of State Owner"] = combined_df.apply(fill_mobile_home_indicator, axis=1)
    else:
        print("Warning: 'Out of State Owner' column not found. Skipping null value handling.")

    # Save the final merged DataFrame to a new CSV file
    print(f"Saving combined data to {output_file}...")
    combined_df.to_csv(output_file, index=False)

    print("Merge complete! Combined data saved successfully.")

if __name__ == "__main__":
    # File paths
    ownership_file = "(2) WebScraping/OwnershipDurations.csv"
    latapult_file = "(1) LatapultData/AllLatapultData.csv"
    output_file = "(3) TransformedData/OwnershipLatapultMerged.csv"

    # List of columns to remove
    columns_to_remove = ["Owner Name", "External Link", "Address", "Owner Address", "Property Address",
                         "Owner 1 Last Name", "Owner 1 First Name & MI", "Owner 2 Last Name", "Owner 2 First Name & MI",
                         "Subdivision Tract Number", "Legal", "Lender Name", "Subdivision Name", "Owner Name", 
                         "External Link", "Address", "Document Number", "Property Address", "Owner Address", 
                         "Subdivision Plat Book", "Subdivision Plat Page", "Sale Book and Page", "Search Index", 
                         "FIPS Code", "Recording Date", "Sale Date", "Mortgage Date", "Mortgage Due Date", 
                         "Mail Direction", "Mail Street Name", "Mail Street Type",
                        "Mail Quadrant", "Mail Unit Number", "Mail City", "Mail State", "Mail Zip Code",  "Subdivision Tract Number", 
                        "Mortgage Deed Type", "Mortgage Term Code", "Mortgage Term", "Sale Code", "Seller Name", "Transaction Type",
                        "Title Company Name", "Last Renovation", "Bedrooms", "Rooms", "Total Baths Calculated", "Total Baths", "Full Baths",
                        "Half Baths", "1 Quarter Baths", "3 Quarter Baths", "Bath Fixtures", "Air Conditioning",
                        "Basement Type", "Building Code", "Building Improvement Code", "Condition", "Construction Type",
                        "Exterior Walls", "Fireplace Number", "Fireplace Type", "Foundation", "Floor", "Frame", "Garage",
                        "Heating", "Parking Spaces", "Parking Type", "Pool", "Construction Quality", "Roof Cover",
                        "Roof Type", "Stories Number", "Total Buildings", "Total Units", "Energy Use", "Fuel", "Sewer",
                        "Water", "GeothinQ Region", "MSA", "Data Year Quarter", "Attribute Vintage", "X", "Y", 
                        "Latitude", "Longitude", "Mail House Number", "Block Number", "Building Square Feet Indicator", "Lender Name", "Lot Number", 
                        "2nd Mortgage Deed Type", "Document Type", "Assessed Improvement Value Ratio", "County", "Appraised Total Value", 
                        "Ground Floor Square Feet", "Gross Square Feet", "Adjusted Gross Square Feet", "Basement Square Feet", "Garage/Parking Square Feet", 
                        "Zip Code", "Deeded Acres", "Zoning Code", "Land Use", "City", "Universal Building Square Feet",
                        "Appraised Land Value", "Appraised Improvement Value", "Front Footage", "Depth Footage", "Building Square Feet",
                        "Living Square Feet", "Assessed Year", "Tax Year", "Assessed Total Value", "Market Total Value", "Mortgage Assumption Amount",
                        "2nd Mortgage Amount", "1st Mortgage Amount", "Market Land Value", "Land Value Calculated", "Mobile Home Indicator",
                        "Improvement Value Calculated", "Market Improvement Value", "Land Square Footage", "Tax Code Area", "Out of State Owner",
                        "Property Indicator", "Owner Corporate Indicator", "State", "Zoning Type"]

    # Ownership type to filter out (set to None if no filtering is needed)
    filter_ownership_type = "current"

    # Property indicator values to filter out (set to None if no filtering is needed)
    filter_property_indicator = ["CONDOMINIUM", "APARTMENT", "AMUSEMENT-RECREATION", "COMMERCIAL", "EXEMPT", "SERVICE", "INDUSTRIAL", 
                                 "WAREHOUSE", "OFFICE BUILDING", "HOSPITAL", "RETAIL", "UTILITIES"]

    # List of columns to remove from ownership data
    ownership_columns_to_remove = ["Owner"]

    final_dropped_columns = ["Duration Type", "Year Built"]

    # Execute the merge function
    append_latapult_to_ownership(
        ownership_file, 
        latapult_file, 
        output_file, 
        columns_to_remove, 
        ownership_columns_to_remove, 
        final_dropped_columns,
        filter_ownership_type, 
        filter_property_indicator,
    )