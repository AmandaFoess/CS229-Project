import pandas as pd

# Path to your CSV file
csv_file = "LatapultData/latapult_results (2).csv"

# Read the CSV file
data = pd.read_csv(csv_file)

# Extract the "Parcel Number" column
parcel_numbers = data["Parcel Number"].dropna().astype(str).str.replace("-", "").unique().tolist()  # Remove NaNs and get unique values

# Display the parcel numbers
print(f"Extracted {len(parcel_numbers)} parcel numbers:")
#print(parcel_numbers)

# Optionally, save the list to a new CSV or text file
pd.DataFrame({"Parcel Number": parcel_numbers}).to_csv("LatapultData/pn_latapult_results (2).csv", index=False)