import yaml
import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import skysurvey

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "Modules"))

from lightcurves import process_lightcurve_data_rates
from plotting import plot_survey_overview

def main():
    print("main() function is starting...")

    # Path to the folder containing the parquet files
    HCorLC = "HC_"
    folder_path = HCorLC + "Lightcurves/lightcurves"

    # Find all .parquet files in the folder
    text_files = glob.glob(os.path.join(folder_path, "*.parquet"))

    num_years = len(text_files)
    if num_years == 0:
        print("No parquet files found.")
        return

    # Parameters for the table
    min_sn_points_values = [1, 2, 3, 4, 5]
    min_detections_values = [1, 2, 3, 4, 5]

    # Create empty tables to store mean and standard deviation of dataset lengths
    lengths_mean_table = pd.DataFrame(index=min_sn_points_values, columns=min_detections_values)
    lengths_std_table = pd.DataFrame(index=min_sn_points_values, columns=min_detections_values)

    # Process each parquet file individually
    for min_sn_points in min_sn_points_values:
        for min_detections in min_detections_values:
            rates_per_file = []

            for file in text_files:
                try:
                    # Read the parquet file
                    dataset = skysurvey.DataSet.read_parquet(file)
                    lightcurves = process_lightcurve_data_rates(
                        dataset, min_sn_points=min_sn_points, min_detections=min_detections
                    )

                    # Count unique lightcurve IDs
                    num_lightcurves = len(lightcurves.data.index.get_level_values(0).unique().tolist())
                    rates_per_file.append(num_lightcurves)

                except Exception as e:
                    print(f"Error processing file {file}: {e}")

            # Calculate mean and standard deviation of the results across all files
            if rates_per_file:
                mean_value = round(np.mean(rates_per_file), 1)
                std_value = round(np.std(rates_per_file), 1)
            else:
                mean_value = 0
                std_value = 0

            # Store results in tables
            lengths_mean_table.loc[min_sn_points, min_detections] = mean_value
            lengths_std_table.loc[min_sn_points, min_detections] = std_value
            print(f"Processed min_sn_points={min_sn_points}, min_detections={min_detections}")

    # Print the results
    print("Mean Table of Lightcurve Counts per Year:")
    print(lengths_mean_table)

    print("\nStandard Deviation Table of Lightcurve Counts per Year:")
    print(lengths_std_table)

    # Function to save tables as CSV files
    def save_table_as_csv(mean_table, std_table, filename):
        combined_table = mean_table.copy()
        
        # Combine mean and standard deviation in the table format
        for col in mean_table.columns:
            combined_table[col] = mean_table[col].astype(str) + " ± " + std_table[col].astype(str)

        # Save as CSV
        combined_table.to_csv(HCorLC + "Lightcurves/" + filename, index=True)
        print(f"Table saved as {filename}")
    # Save tables to CSV
    save_table_as_csv(lengths_mean_table, lengths_std_table, HCorLC+"rates_table.csv")

    print("The rates tables have been saved.")


# Save the table as a PDF with mean and standard deviation
    with PdfPages(HCorLC + "Lightcurves/" + HCorLC + "rates_table.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')

        # Format table values with mean ± std
        table_data = [[f"{lengths_mean_table.loc[i, j]} ± {lengths_std_table.loc[i, j]}"
                       for j in lengths_mean_table.columns] for i in lengths_mean_table.index]

        plt.table(
            cellText=table_data,
            rowLabels=lengths_mean_table.index,
            colLabels=lengths_mean_table.columns,
            loc='center',
            cellLoc='center'
        )

        # Add axis labels and title
        ax.set_title("Mean observation rate by σ threshold and minimal detections", pad=20)
        plt.figtext(0.5, 0.6, "detections", ha="center", fontsize=10)
        plt.figtext(0.08, 0.5, "sigma", va="center", rotation="vertical", fontsize=10)
        plt.figtext(0.5, 0.7, "How many visible SNeIa per year?", ha="center", fontsize=10)

        pdf.savefig(fig)
        plt.close()

    print("The rates table pdf has been created and saved.")


    # Create plot for rates with error bars
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 18})
    colors = ['red', 'blue', 'green', 'purple', 'orange', "pink", "brown"]  # Colors for the lines

    for idx, row in lengths_mean_table.iterrows():
        mean_values = row.astype(float)
        std_values = lengths_std_table.loc[idx].astype(float)
        plt.errorbar(
            lengths_mean_table.columns.astype(int),
            mean_values,
            yerr=std_values,
            label=str(idx) + "-σ threshold",
            capsize=5,
            fmt='-o',  # Line with error bars, no markers
            color=colors[idx - 1]
        )

    plt.xlabel('Minimal number of data points over σ threshold')
    plt.ylabel('Number of SNIa observations per year')
    plt.ylim(0, 1600)
    #plt.title('Mean observation rates regarding the σ thresholding')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.xticks(lengths_mean_table.columns.astype(int), lengths_mean_table.columns)
    plt.tight_layout()
    plt.savefig(HCorLC + "Lightcurves/" + HCorLC + "rates_plot.pdf")

    print("The rates plot has been created and saved.")

#_________________________________-

    # REDSHIFT BINS
    print("Starting redshift binning analysis...")

    # Define bins
    bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]

    # Create bin labels that describe the range
    bin_labels = [
        f"{str(f'{bins[i]:.2f}')[2:]}_to_{str(f'{bins[i+1]:.2f}')[2:]}" for i in range(len(bins) - 1)
    ]

    # Parameters for the table
    min_sn_points_values = [1, 2, 3, 4, 5]

    # Create tables to store mean and standard deviation
    formatted_table_c_mean = pd.DataFrame(index=min_sn_points_values, columns=bin_labels)
    formatted_table_c_std = pd.DataFrame(index=min_sn_points_values, columns=bin_labels)
    formatted_table_n_mean = pd.DataFrame(index=min_sn_points_values, columns=bin_labels)
    formatted_table_n_std = pd.DataFrame(index=min_sn_points_values, columns=bin_labels)

    # Iterate through min_sn_points values
    for min_sn_points in min_sn_points_values:
        c_values_per_bin = {bin_label: [] for bin_label in bin_labels}
        n_values_per_bin = {bin_label: [] for bin_label in bin_labels}

        # Process each parquet file individually
        for file in text_files:
            try:
                dataset = skysurvey.DataSet.read_parquet(file)
                dataset_df = dataset.data.copy()

                # Create a new column for bins
                dataset_df['z_bin'] = pd.cut(dataset_df['z'], bins=bins, labels=bin_labels, right=False)

                # Check if the Parquet file contributes to any bin
                if dataset_df['z_bin'].notna().sum() == 0:
                    print(f"Skipping file {file} - No data for any redshift bin.")
                    continue

                # Group by bins and process each bin separately
                for bin_label, group in dataset_df.groupby('z_bin', observed=False):
                    if bin_label in bin_labels and not group.empty:
                        dataset.set_data(group.drop(columns='z_bin'))
                        lightcurves = process_lightcurve_data_rates(dataset, min_sn_points=min_sn_points, min_detections=3)

                        # Only include files that have data for this bin
                        if not lightcurves.data.empty:
                            c_values_per_bin[bin_label].append(lightcurves.data["c"].mean())
                            n_values_per_bin[bin_label].append(len(lightcurves.data.index.get_level_values(0).unique().tolist()))

            except Exception as e:
                print(f"Error processing file {file}: {e}")

        # Populate the tables with calculated values for this min_sn_points
        for bin_label in bin_labels:
            if c_values_per_bin[bin_label]:  # Check if valid data exists for the bin
                formatted_table_c_mean.loc[min_sn_points, bin_label] = round(np.mean(c_values_per_bin[bin_label]), 3)
                formatted_table_c_std.loc[min_sn_points, bin_label] = round(np.std(c_values_per_bin[bin_label]), 3)
            else:
                formatted_table_c_mean.loc[min_sn_points, bin_label] = -999
                formatted_table_c_std.loc[min_sn_points, bin_label] = -999

            if n_values_per_bin[bin_label]:
                formatted_table_n_mean.loc[min_sn_points, bin_label] = round(np.mean(n_values_per_bin[bin_label]), 1)
                formatted_table_n_std.loc[min_sn_points, bin_label] = round(np.std(n_values_per_bin[bin_label]), 1)
            else:
                formatted_table_n_mean.loc[min_sn_points, bin_label] = 0
                formatted_table_n_std.loc[min_sn_points, bin_label] = 999

    print("The Redshift bin tables have been saved.")

    # Create and save plots as stair plots (without error bars)
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 18})
    colors = ['red', 'blue', 'green', 'purple', 'orange', "pink", "brown"]

    for idx, min_sn_points in enumerate(min_sn_points_values):
        plt.stairs(
            values=pd.to_numeric(formatted_table_c_mean.loc[min_sn_points], errors='coerce').fillna(0).values,
            edges=bins,
            alpha=1,
            color=colors[idx],
            fill=False,
            label=f"{min_sn_points}-σ threshold",
            baseline=None,
            linewidth=3
        )
    plt.xlim(0, 0.13)
    plt.ylim(-0.3, 0.13)
    plt.xlabel('Redshift Bins')
    plt.ylabel('Mean value for color parameter c')
    #plt.title('Binned reddening parameter c as a function of redshift')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(HCorLC + "Lightcurves/" + HCorLC + "redshiftbin_c_plot.pdf")
    print("The reddening plot has been created and saved.")

    # Create stair plot for number of observations
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 18})
    for idx, min_sn_points in enumerate(min_sn_points_values):
        plt.stairs(
            values=pd.to_numeric(formatted_table_n_mean.loc[min_sn_points], errors='coerce').fillna(0).values,
            edges=bins,
            alpha=1,
            color=colors[idx],
            fill=False,
            label=f"{min_sn_points}-σ threshold",
            baseline=None,
            linewidth=3
        )
    plt.xlim(0, 0.13)
    plt.xlabel('Redshift Bins')
    plt.ylabel('Number of SNIa observations per year')
    #plt.title('Binned observation count as a function of redshift')
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(HCorLC + "Lightcurves/" + HCorLC + "redshiftbin_n_plot.pdf")
    print("The rates bin plot has been created and saved.")

    # Save the reddening parameter tables
    save_table_as_csv(formatted_table_c_mean, formatted_table_c_std, "redshiftbin_c_table.csv")

    # Save the observation rates tables
    save_table_as_csv(formatted_table_n_mean, formatted_table_n_std, "redshiftbin_n_table.csv")

    print("CSV files saved successfully.")










    



if __name__ == "__main__":
    main()