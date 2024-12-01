""" import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = '(2) WebScraping/TransferID_MeckArcGISSales.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Ensure 'saledate' is in datetime format
data['saledate'] = pd.to_datetime(data['saledate'])

# Extract year from 'saledate'
data['year'] = data['saledate'].dt.year

# Filter data: Limit Acres to 3
data = data[data['Acres'] <= 0.75]

# Set the style for seaborn
sns.set(style="whitegrid")

# Initialize the plot
plt.figure(figsize=(12, 8))

# Use seaborn's scatterplot to show distributions
sns.scatterplot(
    data=data,
    x='Acres',
    y='VDL Sale Price',
    hue='year',
    palette='tab10',
    alpha=0.6,
    legend='full'
)

# Use seaborn's lineplot to add a trend line (average per year)
sns.lineplot(
    data=data,
    x='Acres',
    y='VDL Sale Price',
    hue='year',
    palette='tab10',
    estimator='mean',  # Calculate the average for each Acres value
    ci=None,           # Turn off confidence intervals
    legend=False
)

# Add plot details
plt.title('Distribution and Average Trends of Acres vs. VDL Sale Price (All Years)', fontsize=16)
plt.xlabel('Acres', fontsize=14)
plt.ylabel('VDL Sale Price ($)', fontsize=14)
plt.grid(True)
plt.legend(title='Year')
plt.tight_layout()

# Save the plot as a PNG file
plot_file = 'acres_vs_sales_avg_trend.png'
plt.savefig(plot_file)
print(f"Plot saved: {plot_file}")

# Show the plot
plt.show()
 """

""" import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = '(2) WebScraping/TransferID_MeckArcGISSales.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Ensure 'saledate' is in datetime format
data['saledate'] = pd.to_datetime(data['saledate'])

# Extract year from 'saledate'
data['year'] = data['saledate'].dt.year

# Filter data: Limit Acres to 0.75
data = data[data['Acres'] <= 0.75]

# Remove outliers in 'VDL Sale Price' using IQR
Q1 = data['VDL Sale Price'].quantile(0.25)
Q3 = data['VDL Sale Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
data = data[(data['VDL Sale Price'] >= lower_bound) & (data['VDL Sale Price'] <= upper_bound)]

# Set the style for seaborn
sns.set(style="whitegrid")

# Initialize the plot
plt.figure(figsize=(12, 8))

# Use seaborn's scatterplot to show distributions
sns.scatterplot(
    data=data,
    x='Acres',
    y='VDL Sale Price',
    hue='year',
    palette='tab10',
    alpha=0.6,
    legend='full'
) 

# Use seaborn's lineplot to add a trend line (average per year)
sns.lineplot(
    data=data,
    x='Acres',
    y='VDL Sale Price',
    hue='year',
    palette='tab10',
    estimator='mean',  # Calculate the average for each Acres value
    ci=None,           # Turn off confidence intervals
    legend=False
)

# Add plot details
plt.title('Distribution and Average Trends of Acres vs. VDL Sale Price (All Years)', fontsize=16)
plt.xlabel('Acres', fontsize=14)
plt.ylabel('VDL Sale Price ($)', fontsize=14)
plt.grid(True)
plt.legend(title='Year')
plt.tight_layout()

# Save the plot as a PNG file
plot_file = 'acres_vs_sales_avg_trend.png'
plt.savefig(plot_file)
print(f"Plot saved: {plot_file}")

# Show the plot
plt.show() """

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = '(2) WebScraping/TransferID_MeckArcGISSales.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Ensure 'saledate' is in datetime format
data['saledate'] = pd.to_datetime(data['saledate'])

# Extract year from 'saledate'
data['year'] = data['saledate'].dt.year

# Filter data: Limit Acres to 0.75
data = data[data['Acres'] <= 0.75]

# Remove outliers in 'VDL Sale Price' using IQR
Q1 = data['VDL Sale Price'].quantile(0.25)
Q3 = data['VDL Sale Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['VDL Sale Price'] >= lower_bound) & (data['VDL Sale Price'] <= upper_bound)]

# Set the style for seaborn
sns.set(style="whitegrid")

# Initialize the plot
plt.figure(figsize=(12, 8))

# Bin the Acres values to smooth the trend line
bins = np.linspace(0, 0.75, 15)  # Create 15 bins between 0 and 0.75
data['Acres_Binned'] = pd.cut(data['Acres'], bins=bins, labels=bins[:-1])

# Aggregate data by binned Acres and calculate mean VDL Sale Price
binned_data = data.groupby(['Acres_Binned', 'year']).agg({'VDL Sale Price': 'mean'}).reset_index()
binned_data['Acres_Binned'] = binned_data['Acres_Binned'].astype(float)

# Use seaborn's lineplot for the smoothed trend line
sns.lineplot(
    data=binned_data,
    x='Acres_Binned',
    y='VDL Sale Price',
    hue='year',
    palette='tab10',
    ci=None,
    legend='full'
)

# Add plot details
plt.title('Distribution and Smoothed Trends of Acres vs. VDL Sale Price (All Years)', fontsize=16)
plt.xlabel('Acres', fontsize=14)
plt.ylabel('VDL Sale Price ($)', fontsize=14)
plt.grid(True)
plt.legend(title='Year')
plt.tight_layout()

# Save the plot as a PNG file
plot_file = 'acres_vs_sales_smoothed_trend.png'
plt.savefig(plot_file)
print(f"Plot saved: {plot_file}")

# Show the plot
plt.show()
