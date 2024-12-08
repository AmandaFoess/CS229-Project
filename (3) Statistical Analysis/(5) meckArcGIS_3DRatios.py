# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import binned_statistic_2d
# from scipy.ndimage import gaussian_filter
# from matplotlib.colors import ListedColormap, BoundaryNorm

# # Load the dataset
# input_csv = "(3) TransformedData/Addresses_with_Ratio.csv"
# df = pd.read_csv(input_csv)

# # Define the features to visualize
# features = ["Latitude", "Longitude", "House to Lot Ratio"]

# # Filter out records with missing values in the required features
# df = df.dropna(subset=features)

# # Outlier removal using IQR
# def remove_outliers(df, column, range):
#     """Removes outliers based on the IQR method."""
#     Q1 = df[column].quantile(range)
#     Q3 = df[column].quantile(1 - range)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# # Remove outliers from relevant columns
# df = remove_outliers(df, "House to Lot Ratio", 0.25)
# df = remove_outliers(df, "Longitude", 0.05)
# df = remove_outliers(df, "Latitude", 0.05)

# # Extract the cleaned features for plotting
# latitudes = df["Latitude"]
# longitudes = df["Longitude"]
# ratios = df["House to Lot Ratio"]

# # Create a binned statistic for average ratio
# stat, x_edges, y_edges, _ = binned_statistic_2d(
#     longitudes, latitudes, ratios, statistic="mean", bins=50
# )

# # Replace NaN values in the stat array with 0
# stat = np.nan_to_num(stat, nan=0)

# # Apply Gaussian smoothing to the stat array for smoother transitions
# smoothed_stat = gaussian_filter(stat, sigma=1.1)

# # Calculate bin centers for plotting
# x_bin_centers = (x_edges[:-1] + x_edges[1:]) / 2
# y_bin_centers = (y_edges[:-1] + y_edges[1:]) / 2
# X, Y = np.meshgrid(x_bin_centers, y_bin_centers)

# # Define 10 color ranges
# max_value = np.nanmax(smoothed_stat)
# bounds = np.linspace(0, max_value, 10)  # Create 10 intervals
# colors = [
#     (1.0, 1.0, 1.0),  # White for 0
#     (0.5, 0.0, 0.5),  # Purple
#     (0.7, 0.0, 0.3),  # Magenta
#     (1.0, 0.5, 0.0),  # Orange
#     (1.0, 1.0, 0.0),  # Yellow
#     (0.8, 1.0, 0.4),  # Light green
#     (0.0, 1.0, 0.0),  # Green
#     (0.0, 0.8, 0.8),  # Teal
#     (0.0, 0.0, 1.0),  # Blue
# ]
# cmap = ListedColormap(colors)

# # Define boundaries and normalization
# norm = BoundaryNorm(boundaries=bounds, ncolors=len(colors))

# # Plot the filled contour
# plt.figure(figsize=(12, 8))
# ax = plt.gca()
# ax.set_facecolor('white')  # Ensure background remains white

# contourf = plt.contourf(
#     X, Y, smoothed_stat.T, levels=bounds, cmap=cmap, norm=norm
# )
# contour_lines = plt.contour(
#     X, Y, smoothed_stat.T, levels=bounds, colors='black', linewidths=0.5
# )

# # Add color bar
# cbar = plt.colorbar(contourf, boundaries=bounds, ticks=bounds, spacing='proportional')
# cbar.set_label("Average House to Lot Ratio")

# # Add labels and title
# plt.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")
# plt.title("Smoothed Topographic Contour Plot with 10 Color Ranges")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")

# plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import binned_statistic_2d
# from scipy.ndimage import gaussian_filter
# import geopandas as gpd
# import contextily as ctx
# from matplotlib.colors import ListedColormap, BoundaryNorm

# # Load the dataset
# input_csv = "(3) TransformedData/Addresses_with_Ratio.csv"
# df = pd.read_csv(input_csv)

# # Define the features to visualize
# features = ["Latitude", "Longitude", "House to Lot Ratio"]

# # Filter out records with missing values in the required features
# df = df.dropna(subset=features)

# # Outlier removal using IQR
# def remove_outliers(df, column, quantile_range):
#     """Removes outliers based on the IQR method."""
#     Q1 = df[column].quantile(quantile_range)
#     Q3 = df[column].quantile(1 - quantile_range)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# # Remove outliers from relevant columns
# df = remove_outliers(df, "House to Lot Ratio", 0.25)
# df = remove_outliers(df, "Longitude", 0.05)
# df = remove_outliers(df, "Latitude", 0.05)

# # Create a GeoDataFrame and set CRS to WGS84
# gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]))
# gdf.crs = "EPSG:4326"

# # Reproject to Web Mercator
# gdf = gdf.to_crs(epsg=3857)

# # Extract projected coordinates
# df["x"] = gdf.geometry.x
# df["y"] = gdf.geometry.y

# # Create a binned statistic for average ratio
# stat, x_edges, y_edges, _ = binned_statistic_2d(
#     df["x"], df["y"], df["House to Lot Ratio"], statistic="mean", bins=50
# )

# # Replace NaN values in the stat array with 0
# stat = np.nan_to_num(stat, nan=0)

# # Apply Gaussian smoothing to the stat array for smoother transitions
# smoothed_stat = gaussian_filter(stat, sigma=1)

# # Calculate bin centers for plotting
# x_bin_centers = (x_edges[:-1] + x_edges[1:]) / 2
# y_bin_centers = (y_edges[:-1] + y_edges[1:]) / 2
# X, Y = np.meshgrid(x_bin_centers, y_bin_centers)

# # Define color ranges
# bounds = np.linspace(0, np.nanmax(smoothed_stat), 11)  # 10 intervals
# colors = [
#     (1.0, 1.0, 1.0),  # White for 0
#     (0.5, 0.0, 0.5),  # Purple
#     (0.7, 0.0, 0.3),  # Magenta
#     (1.0, 0.5, 0.0),  # Orange
#     (1.0, 1.0, 0.0),  # Yellow
#     (0.8, 1.0, 0.4),  # Light green
#     (0.0, 1.0, 0.0),  # Green
#     (0.0, 0.8, 0.8),  # Teal
#     (0.0, 0.0, 1.0),  # Blue
#     (0.5, 0.0, 0.5),  # Purple
# ]
# cmap = ListedColormap(colors)
# norm = BoundaryNorm(boundaries=bounds, ncolors=len(colors))

# # Plot the data with a map in the background
# fig, ax = plt.subplots(figsize=(12, 8))

# # Overlay the filled contour
# contourf = ax.contourf(
#     X, Y, smoothed_stat.T, levels=bounds, cmap=cmap, norm=norm, alpha=0.3
# )

# # Add contour lines
# contour_lines = ax.contour(
#     X, Y, smoothed_stat.T, levels=bounds, colors="black", linewidths=0.5
# )

# # Add color bar
# cbar = plt.colorbar(contourf, boundaries=bounds, ticks=bounds, spacing="proportional")
# cbar.set_label("Average House to Lot Ratio")

# # Add basemap using OpenStreetMap Mapnik tiles
# ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, alpha=1)

# # Add labels and title
# plt.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")
# plt.title("Smoothed Topographic Contour Plot with Basemap")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")

# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
import geopandas as gpd
import contextily as ctx
from matplotlib.colors import ListedColormap, BoundaryNorm
import os

# Load the dataset
input_csv = "(3) TransformedData/Addresses_with_Mapped_Grantees_Cleaned.csv"
df = pd.read_csv(input_csv)

# Extract year from the saledate column
df['Year'] = pd.to_datetime(df['saledate']).dt.year

# Define the features to visualize
features = ["Latitude", "Longitude", "House to Lot Ratio", "Year"]

# Filter out records with missing values in the required features
df = df.dropna(subset=features)

# Outlier removal using IQR
def remove_outliers(df, column, quantile_range):
    """Removes outliers based on the IQR method."""
    Q1 = df[column].quantile(quantile_range)
    Q3 = df[column].quantile(1 - quantile_range)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from relevant columns
df = remove_outliers(df, "House to Lot Ratio", 0.25)
df = remove_outliers(df, "Longitude", 0.05)
df = remove_outliers(df, "Latitude", 0.05)

# Create a GeoDataFrame and set CRS to WGS84
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]))
gdf.crs = "EPSG:4326"

# Reproject to Web Mercator
gdf = gdf.to_crs(epsg=3857)

# Extract projected coordinates
df["x"] = gdf.geometry.x
df["y"] = gdf.geometry.y

# Group data by year
years = df["Year"].unique()
output_dir = "Yearly_Plots"
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

for year in sorted(years):
    df_year = df[df["Year"] == year]

    # Create a binned statistic for average ratio
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        df_year["x"], df_year["y"], df_year["House to Lot Ratio"], statistic="mean", bins=50
    )

    # Replace NaN values in the stat array with 0
    stat = np.nan_to_num(stat, nan=0)

    # Apply Gaussian smoothing to the stat array for smoother transitions
    smoothed_stat = gaussian_filter(stat, sigma=0.75)

    # Calculate bin centers for plotting
    x_bin_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_bin_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_bin_centers, y_bin_centers)

    # Define color ranges
    bounds = np.linspace(0, np.nanmax(smoothed_stat), 11)  # 10 intervals
    colors = [
        (1.0, 1.0, 1.0),  # White for 0
        (1.0, 0.0, 0.0),  # Red for low values
        (1.0, 0.5, 0.0),  # Orange
        (1.0, 1.0, 0.0),  # Yellow
        (0.5, 1.0, 0.5),  # Light Green
        (0.0, 1.0, 0.0),  # Green
        (0.0, 1.0, 1.0),  # Cyan
        (0.0, 0.5, 1.0),  # Light Blue
        (0.0, 0.0, 1.0),  # Blue
        (0.5, 0.0, 1.0),  # Purple
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=bounds, ncolors=len(colors))

    # Plot the data with a map in the background
    fig, ax = plt.subplots(figsize=(12, 8))

    # Overlay the filled contour
    contourf = ax.contourf(
        X, Y, smoothed_stat.T, levels=bounds, cmap=cmap, norm=norm, alpha=0.3
    )

    # Add contour lines
    contour_lines = ax.contour(
        X, Y, smoothed_stat.T, levels=bounds, colors="black", linewidths=0.5
    )

    # Add color bar
    cbar = plt.colorbar(contourf, boundaries=bounds, ticks=bounds, spacing="proportional")
    cbar.set_label("Average House to Lot Ratio")

    # Add basemap using OpenStreetMap Mapnik tiles
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, alpha=1)

    # Add labels and title
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")
    plt.title(f"Smoothed Topographic Contour Plot for {year}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Save the plot to a file
    output_file = os.path.join(output_dir, f"House_to_Lot_Ratio_{year}.png")
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    print(f"Saved plot for {year} to {output_file}")

print("All yearly plots have been created.")
