import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tabulate import tabulate
import pdfkit
import subprocess
from pdflatex import PDFLaTeX

# Define a function to analyze a transformed dataset
def analyze_transformed_dataset(file_path, output_summary_path):
    try:
        # Load the transformed dataset
        df = pd.read_csv(file_path)

        # Create a dictionary to store summary statistics
        summary = {}

        # Numerical Features Analysis
        numerical_features = df.select_dtypes(include=['float64', 'int64']).columns

        for feature in numerical_features:
            # Central Tendency
            # Check for NaN values and handle them
            if df[feature].notna().sum() > 0:  # Check if the column has valid values
                mean = df[feature].mean()
                median = df[feature].median()
                mode = df[feature].mode()[0] if not df[feature].mode().empty else np.nan
                # Dispersion
                std_dev = df[feature].std()
                variance = df[feature].var()
                range_val = df[feature].max() - df[feature].min()
                iqr = df[feature].quantile(0.75) - df[feature].quantile(0.25)
            else:
                # If all values are NaN, set statistics to NaN and print a warning
                mean = median = mode = std_dev = variance = range_val = iqr = np.nan
                print(f"Warning: Feature '{feature}' contains only NaN values. Statistics set to NaN.")

            # Store in summary dictionary
            summary[feature] = {
                "Mean": round(mean, 3),
                "Median": round(median, 3),
                "Mode": round(mode, 3),
                "Std Dev": round(std_dev, 3),
                "Variance": round(variance, 3),
                "Range": round(range_val, 3),
                "IQR": round(iqr, 3)
            }

        # Save summary to a CSV file
        summary_df = pd.DataFrame.from_dict(summary, orient='index')
        summary_df.to_csv(output_summary_path)
        print(f"Summary statistics saved to {output_summary_path}")

        # Distribution Analysis
        output_plots_dir = "plots"
        os.makedirs(output_plots_dir, exist_ok=True)

        for feature in numerical_features:
            plt.figure(figsize=(8, 6))

            # Histogram
            sns.histplot(df[feature], kde=True, bins=30, color="blue")
            plt.title(f"Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")

            # Save the plot
            plot_path = f"{output_plots_dir}/{feature}_distribution.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved distribution plot for {feature} to {plot_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def csv_to_pretty_table_pdf(csv_file, output_pdf):

    # Configure pdfkit with wkhtmltopdf path
    config = pdfkit.configuration(wkhtmltopdf='/opt/miniconda3/envs/SellerWillingness/bin/wkhtmltopdf')

    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)

        # Format numerical columns: smaller sigfigs and scientific notation
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = df[col].apply(lambda x: f"{x:.2e}" if not pd.isna(x) else x)  # 3 significant figures in scientific notation

        # Generate the tabulate table in LaTeX format
        table_latex = tabulate(df, headers='keys', tablefmt='latex', showindex=False)

        # Add LaTeX boilerplate to compile it to a PDF
        latex_boilerplate = f"""
        \\documentclass[a4paper,10pt]{{article}}
        \\usepackage[utf8]{{inputenc}}
        \\usepackage{{array}}
        \\usepackage{{booktabs}}
        \\usepackage{{longtable}}
        \\usepackage{{geometry}}
        \\geometry{{a4paper, margin=1in}}
        \\begin{{document}}
        \\section*{{Summary Table}}
        {table_latex}
        \\end{{document}}
        """

        # Save the LaTeX content to a temporary file
        temp_tex_file = "temp_table.tex"
        with open(temp_tex_file, "w") as tex_file:
            tex_file.write(latex_boilerplate)

        # Compile the LaTeX file to PDF using pdflatex
        subprocess.run(["pdflatex", temp_tex_file], check=True)

        # Move the generated PDF to the desired output location
        temp_pdf_file = temp_tex_file.replace(".tex", ".pdf")
        os.rename(temp_pdf_file, output_pdf)

        # Clean up auxiliary files
        for ext in [".aux", ".log", ".out"]:
            aux_file = temp_tex_file.replace(".tex", ext)
            if os.path.exists(aux_file):
                os.remove(aux_file)

        print(f"Table saved as a PDF: {output_pdf}")

    except subprocess.CalledProcessError as e:
        print(f"Error compiling LaTeX to PDF: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Analyze Features
    transformed_file = "TransformedData/distance_latapult_results (2).csv"  # Replace with your transformed file path
    output_summary_file = "StatisticalAnalysis/distance_latapult_results (2).csv"
    analyze_transformed_dataset(transformed_file, output_summary_file)

    # Print Statistical Summaries
    output_pdf = "StatisticalAnalysis/distance_latapult_results (2).pdf"
    csv_to_pretty_table_pdf(output_summary_file, output_pdf)