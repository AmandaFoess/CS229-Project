import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_file, train_output_file, test_output_file, test_size=0.2):
    """
    Splits the input dataset into training and testing datasets and saves them to separate files.
    
    Parameters:
    - input_file (str): Path to the input CSV file.
    - train_output_file (str): Path to save the training dataset.
    - test_output_file (str): Path to save the testing dataset.
    - test_size (float): Proportion of the data to include in the test set. Default is 0.2 (20%).
    """
    try:
        # Load the dataset
        df = pd.read_csv(input_file)
        
        # Ensure test_size is a valid proportion
        if not (0 < test_size < 1):
            raise ValueError("Test size must be a float between 0 and 1 (e.g., 0.2 for 20%).")
        
        # Split the dataset into training and testing sets
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        # Save the training and testing datasets
        train_df.to_csv(train_output_file, index=False)
        test_df.to_csv(test_output_file, index=False)
        
        print(f"Training dataset saved to {train_output_file} with {len(train_df)} rows.")
        print(f"Testing dataset saved to {test_output_file} with {len(test_df)} rows.")

    except Exception as e:
        print(f"Error splitting dataset: {e}")

# Example Usage
# Specify the input file and the output file paths
input_file_path = "(3) TransformedData/OwnershipLatapultMerged.csv"
train_output_file_path = "(5) Datasets/TrainSet.csv"  # Replace with the desired training file path
test_output_file_path = "(5) Datasets/TestSet.csv"    # Replace with the desired testing file path

# Split the dataset into training and testing sets
split_dataset(input_file_path, train_output_file_path, test_output_file_path, test_size=0.2)
