import pandas as pd

# File path for the results CSV
results_csv = "(3) TransformedData/IQR_Grid_Search_Results.csv"
filtered_results_csv = "(4) StatisticalAnalysis/IQR_Filtered_Results.csv"

# Load the results CSV
try:
    results_df = pd.read_csv(results_csv)
    print("Loaded results dataset successfully.")
except FileNotFoundError:
    print(f"File {results_csv} not found. Ensure the file exists and try again.")
    exit()

# Analyze the results
# Sort by 'R^2 Test' in descending order
results_df = results_df.sort_values(by='R^2 Test', ascending=False)

# Define a threshold for top results
top_threshold = 0.98  # Adjust based on your preference for "high" R^2 values

# Filter for top results
top_results = results_df[results_df['R^2 Test'] >= top_threshold]

# If no rows match the threshold, fallback to selecting the top 10 results
if top_results.empty:
    print(f"No results with 'R^2 Test' >= {top_threshold}. Showing top 10 results instead.")
    top_results = results_df.head(10)

# Display the top results
print("\nTop Results:")
print(top_results)

# Save the top results to a CSV file
top_results_csv = "(3) TransformedData/IQR_Top_Results.csv"
top_results.to_csv(top_results_csv, index=False)
print(f"Top results saved to {top_results_csv}")

# Summarize key insights
print("\nSummary of Key Insights:")
print(f"Total Configurations Tested: {len(results_df)}")
print(f"Total Configurations with 'R^2 Test' >= {top_threshold}: {len(top_results)}")

# Display common configurations in top results
columns_to_analyze = [
    "House to Lot Ratio (Bottom)", "House to Lot Ratio (Top)",
    "Acres (Bottom)", "Acres (Top)",
    "VDL Sale Price (Bottom)", "VDL Sale Price (Top)"
]

for column in columns_to_analyze:
    print(f"\nMost Common Values for '{column}' in Top Results:")
    print(top_results[column].value_counts().head(5))  # Show top 5 most frequent values

# Define a reasonable max value for House to Lot Ratio
max_reasonable_ratio = 0.2

# Filter results where 'House to Lot Ratio' does not exceed the reasonable max
filtered_results_df = results_df[results_df['House to Lot Ratio (Top)'] >= max_reasonable_ratio]

if filtered_results_df.empty:
    print(f"No results found with 'House to Lot Ratio' <= {max_reasonable_ratio}.")
else:
    # Sort by 'R^2 Test' descending and then by 'MSE Test' ascending
    filtered_results_df = filtered_results_df.sort_values(by=['R^2 Test', 'MSE Test'], ascending=[False, True])

    # Save the filtered results to a new CSV
    filtered_results_df.to_csv(filtered_results_csv, index=False)
    print(f"Filtered results saved to {filtered_results_csv}")

    # Display top 10 results
    print("\nTop 10 Filtered Results:")
    print(filtered_results_df.head(10))

    # Summarize most common IQR values for each variable in the top results
    print("\nSummary of Key Insights:")
    for col in ['House to Lot Ratio (Bottom)', 'House to Lot Ratio (Top)', 
                'Acres (Bottom)', 'Acres (Top)', 'VDL Sale Price (Bottom)', 'VDL Sale Price (Top)']:
        common_values = filtered_results_df[col].value_counts().head(5)
        print(f"\nMost Common Values for '{col}' in Top Results:")
        print(common_values)

    # Retrieve the best configuration
    best_config = filtered_results_df.iloc[0]
    print("\nBest Configuration:")
    print(best_config)