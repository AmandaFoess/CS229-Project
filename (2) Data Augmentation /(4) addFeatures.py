import pandas as pd

# File paths
addresses_file = "(3) TransformedData/Addresses_with_Ratio.csv"  # Update path if needed
additional_file = "(2) WebScraping/Mod_MeckArcGISSales_Sorted.csv"  # Update path if needed
output_file = "(3) TransformedData/Addresses_with_Ratios_and_New_Features.csv"  # Update path if needed

# Load the datasets into DataFrames
addresses_df = pd.read_csv(addresses_file)
additional_df = pd.read_csv(additional_file)

# Ensure both datasets have the key columns as strings (to avoid mismatches due to type)
addresses_df['parcelid'] = addresses_df['parcelid'].astype(str)
addresses_df['transferid'] = addresses_df['transferid'].astype(str)
additional_df['parcelid'] = additional_df['parcelid'].astype(str)
additional_df['transferid'] = additional_df['transferid'].astype(str)

# Merge the datasets on 'parcelid' and 'transferid'
merged_df = pd.merge(
    addresses_df,
    additional_df[['parcelid', 'transferid', 'landuseful']],  # Select only the needed column(s)
    on=['parcelid', 'transferid'],  # Merge on parcelid and transferid
    how='left'  # Use 'left' join to retain all rows from addresses_df
)

# Step 1: Filter landuse features based on landuse_dict
landuse_dict = {
    "MEDICAL CONDOMINIUM": "drop",
    "USE VALUE HOMESITE": "keep",
    "CONDOMINIUM": "drop",
    "RURAL HOMESITE": "keep",
    "CHURCH": "drop",
    "SINGLE FAMILY RESIDENTIAL - ACREAGE": "keep",
    "COMMERCIAL": "drop",
    "TOWN HOUSE SFR": "keep",
    "INDUSTRIAL": "drop",
    "AGRICULTURAL - COMMERCIAL PRODUCTION": "drop",
    "GOLF COURSE CLASS 2 - PRIVATE CLUB": "drop",
    "MULTI FAMILY": "keep",
    "SINGLE FAMILY RESIDENTIAL - GOLF": "keep",
    "WAREHOUSING": "drop",
    "OFFICE CONDOMINIUM": "drop",
    "SINGLE FAMILY RESIDENTIAL": "keep",
    "FLUM/SWIM FLOODWAY (NO BUILD ZONE)": "drop",
    "LABORATORY / RESEARCH": "drop",
    "FOREST - COMMERCIAL PRODUCTION": "drop",
    "SINGLE FAMILY RESIDENTIAL - COMMON": "keep",
    "RESERVED PARCEL": "drop",
    "MULTI FAMILY GARDEN": "keep",
    "MOBILE HOME SUBDIVISION": "drop",
    "OFFICE": "drop",
    "WASTELAND, SLIVERS, GULLIES, ROCK OUTCROP": "drop",
    "NEW PARCEL": "drop",
    "100 YEAR FLOOD PLAIN - AC": "drop",
    "GOLF COURSE CLASS 1 - CHAMPIONSHIP": "drop",
    "ROADWAY CORRIDOR": "drop",
    "WAREHOUSE CONDOMINIUM": "drop",
    "SINGLE FAMILY RESIDENTIAL - WATERFRONT": "keep",
    "NO LAND INTEREST": "drop",
    "MULTI FAMILY DUPLEX/TRIPLEX": "keep",
    "MINI WAREHOUSE": "drop",
    "LUMBER YARD": "drop",
    "COMMERCIAL CONDOMINIUM": "drop",
    "MULTI FAMILY MARINA LAND": "keep",
    "AIR RIGHTS PARCEL": "drop",
    "SUBMERGED LAND, RIVERS AND LAKES": "drop",
    "UNSUITABLE FOR SEPTIC": "drop",
    "SCHOOL, COLLEGE, PRIVATE": "keep",
    "BUFFER STRIP": "drop",
    "SINGLE FAMILY RESIDENTIAL - RIVER": "keep"
}

# Step 2: Create a list of landuse features to drop
features_to_drop = [key for key, value in landuse_dict.items() if value == "drop"]

# Step 3: Filter the DataFrame to exclude records with "drop" landuse features
merged_df = merged_df[~merged_df['landuseful'].isin(features_to_drop)]

# Step 4: Remove grantee values that appear only once
grantee_counts = merged_df['grantee'].value_counts()
unique_grantees = grantee_counts[grantee_counts > 1].index
merged_df = merged_df[merged_df['grantee'].isin(unique_grantees)]


# Save the filtered dataset to a new CSV file
merged_df.to_csv(output_file, index=False)
print(f"Merged dataset saved to {output_file}")
