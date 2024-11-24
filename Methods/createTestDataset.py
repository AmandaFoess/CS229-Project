import pandas as pd
import random
import TransformedData.getDistance as getDistance

def create_test_dataset(input_file, output_file, sample_size=50):
    """
    Randomly selects a specified number of rows from a CSV file and saves them to a new file.
    
    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to save the output CSV file with the sampled rows.
    - sample_size (int): Number of rows to sample. Default is 50.
    """
    try:
        # Load the dataset
        df = pd.read_csv(input_file)
        
        # Check if the dataset has enough rows
        if len(df) < sample_size:
            raise ValueError(f"The dataset has only {len(df)} rows, but {sample_size} rows are required.")

        # Randomly select rows
        sampled_df = df.sample(n=sample_size, random_state=42)

        # Save the sampled dataset
        sampled_df.to_csv(output_file, index=False)
        print(f"Test dataset with {sample_size} rows saved to {output_file}")
    
    except Exception as e:
        print(f"Error creating test dataset: {e}")

# Example Usage
# Specify the input file and the output test file
input_file_path = "LatapultData/latapult_results (2).csv"
output_file_path = "TransformedData/large_test_dataset.csv"  # Replace with the desired output file path

# Create the test dataset
create_test_dataset(input_file_path, output_file_path, sample_size=500)

getDistance.transform_dataset(output_file_path)
