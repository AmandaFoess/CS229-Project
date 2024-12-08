import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# File paths
file_path = '(3) TransformedData/TransferID_with_Ratio.csv'  # Input file with House to Lot Ratio
output_file_ratio = 'acres_vs_ratio_smoothed_trend.png'  # Output plot for House to Lot Ratio

# Parameters
max_finished_home_value = 400_000  # Set the maximum Finished Home Value

# Load the dataset
data = pd.read_csv(file_path)

# Ensure 'saledate' is in datetime format
data['saledate'] = pd.to_datetime(data['saledate'])

# Extract year from 'saledate'
data['year'] = data['saledate'].dt.year

# Filter data: Limit Acres to 0.75
data = data[data['Acres'] <= 0.75]

# Filter data: Limit Finished Home Value to max_finished_home_value
data = data[data['Finished Home Value'] <= max_finished_home_value]

# Remove outliers in 'House to Lot Ratio' using IQR
Q1_ratio = data['House to Lot Ratio'].quantile(0.25)
Q3_ratio = data['House to Lot Ratio'].quantile(0.75)
IQR_ratio = Q3_ratio - Q1_ratio
lower_bound_ratio = Q1_ratio - 1.5 * IQR_ratio
upper_bound_ratio = Q3_ratio + 1.5 * IQR_ratio
data = data[(data['House to Lot Ratio'] >= lower_bound_ratio) & (data['House to Lot Ratio'] <= upper_bound_ratio)]

# Set the style for seaborn
sns.set(style="whitegrid")

# Plot: House to Lot Ratio vs Acres
plt.figure(figsize=(12, 8))

# Bin the Acres values to smooth the trend line
bins = np.linspace(0, 0.75, 15)  # Create 15 bins between 0 and 0.75
data['Acres_Binned'] = pd.cut(data['Acres'], bins=bins, labels=bins[:-1])

# Aggregate data by binned Acres and calculate mean House to Lot Ratio
binned_data_ratio = data.groupby(['Acres_Binned', 'year']).agg({'House to Lot Ratio': 'mean'}).reset_index()
binned_data_ratio['Acres_Binned'] = binned_data_ratio['Acres_Binned'].astype(float)

# Use seaborn's lineplot for the smoothed trend line
sns.lineplot(
    data=binned_data_ratio,
    x='Acres_Binned',
    y='House to Lot Ratio',
    hue='year',
    palette='tab10',
    ci=None,
    legend='full'
)

# Add plot details
plt.title('Distribution and Smoothed Trends of Acres vs. House to Lot Ratio (All Years)', fontsize=16)
plt.xlabel('Acres', fontsize=14)
plt.ylabel('House to Lot Ratio', fontsize=14)
plt.grid(True)
plt.legend(title='Year')
plt.tight_layout()

# Save the second plot
plt.savefig(output_file_ratio)
print(f"Plot saved: {output_file_ratio}")

# Show the second plot
plt.show()
