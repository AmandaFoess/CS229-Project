import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ownership_durations(input_csv, output_folder):
    """
    Create plots for ownership duration data and save them to the output folder.

    Args:
        input_csv (str): Path to the CSV file with ownership duration data.
        output_folder (str): Folder to save the plots.
    """
    # Load the data
    df = pd.read_csv(input_csv)

    # Ensure necessary columns are present
    required_columns = ["Parcel Number", "Ownership Duration", "Duration Type", "Owner"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain the columns: {', '.join(required_columns)}")

    # Convert "Ownership Duration" to numeric, handling missing values
    df["Ownership Duration"] = pd.to_numeric(df["Ownership Duration"], errors="coerce")

    # Plot 1: Distribution of Ownership Durations
    plt.figure(figsize=(8, 6))
    sns.histplot(df["Ownership Duration"].dropna(), bins=20, kde=True)
    plt.title("Distribution of Ownership Durations")
    plt.xlabel("Ownership Duration (Years)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig(f"{output_folder}/ownership_duration_distribution.png")
    plt.show()

    # Plot 2: Boxplot of Ownership Durations by Duration Type
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Duration Type", y="Ownership Duration")
    plt.title("Ownership Duration by Duration Type")
    plt.xlabel("Duration Type")
    plt.ylabel("Ownership Duration (Years)")
    plt.grid()
    plt.savefig(f"{output_folder}/ownership_duration_boxplot.png")
    plt.show()

    # Plot 3: Countplot of Duration Types
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="Duration Type")
    plt.title("Count of Duration Types")
    plt.xlabel("Duration Type")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig(f"{output_folder}/duration_type_countplot.png")
    plt.show()

    # Plot 4: Missing Ownership Duration
    plt.figure(figsize=(8, 6))
    missing_counts = df["Ownership Duration"].isnull().value_counts()
    missing_counts.index = ["Not Missing", "Missing"]
    missing_counts.plot(kind="bar", color=["green", "red"])
    plt.title("Missing Ownership Durations")
    plt.xlabel("Missing Status")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig(f"{output_folder}/missing_ownership_duration.png")
    plt.show()

    # Plot 5: Distribution of Ownership Durations by Ownership Type
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x="Ownership Duration",
        hue="Duration Type",
        kde=True,
        bins=20,
        element="step",
        palette="Set2"
    )
    plt.title("Distribution of Ownership Durations by Ownership Type")
    plt.xlabel("Ownership Duration (Years)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig(f"{output_folder}/ownership_duration_distribution_by_type.png")
    plt.show()

    print(f"All plots saved in the folder: {output_folder}")


if __name__ == "__main__":
    # Input file with ownership durations
    input_csv = "(2) WebScraping/OwnershipDurations.csv"

    # Output folder for plots
    output_folder = "(2) WebScraping/Plots"

    # Create plots
    plot_ownership_durations(input_csv, output_folder)