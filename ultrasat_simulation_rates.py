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
from templates import create_salt3_template

def main():
    print("main() function is starting...")

    HCorLC = "LClim_"
    folder_path = HCorLC + "Lightcurves/lightcurves"
    text_files = glob.glob(os.path.join(folder_path, "*.parquet"))
    num_years = len(text_files)
    if num_years == 0:
        print("No parquet files found.")
        return

    min_sn_points_values = [1, 2, 3, 4, 5]
    min_detections_values = [1, 2, 3, 4, 5]

    # ►► NEU: zusätzliches DataFrame für die SEM-Werte
    lengths_mean_table = pd.DataFrame(index=min_sn_points_values, columns=min_detections_values)
    lengths_std_table  = pd.DataFrame(index=min_sn_points_values, columns=min_detections_values)
    lengths_sem_table  = pd.DataFrame(index=min_sn_points_values, columns=min_detections_values)  # NEU

    for min_sn_points in min_sn_points_values:
        for min_detections in min_detections_values:
            rates_per_file = []

            for file in text_files:
                try:
                    dataset = skysurvey.DataSet.read_parquet(file)
                    lightcurves = process_lightcurve_data_rates(
                        dataset,
                        min_sn_points=min_sn_points,
                        min_detections=min_detections
                    )

                    num_lightcurves = len(lightcurves.data.index.get_level_values(0).unique())
                    rates_per_file.append(num_lightcurves)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

            # ►► SD und SEM berechnen
            if rates_per_file:
                mean_value = round(np.mean(rates_per_file), 1)
                std_value  = round(np.std(rates_per_file, ddof=0), 1)
                sem_value  = round(std_value / np.sqrt(len(rates_per_file)), 1)   # NEU
            else:
                mean_value = std_value = sem_value = 0

            lengths_mean_table.loc[min_sn_points, min_detections] = mean_value
            lengths_std_table.loc[min_sn_points,  min_detections] = std_value
            lengths_sem_table.loc[min_sn_points,  min_detections] = sem_value     # NEU
            print(f"Processed min_sn_points={min_sn_points}, min_detections={min_detections}")

    # --- Ausgaben ------------------------------------------------------------------
    print("Mean Table of Lightcurve Counts per Year:")
    print(lengths_mean_table)

    print("\nStandard Deviation Table of Lightcurve Counts per Year:")
    print(lengths_std_table)

    print("\nStandard Error (SEM) Table of Lightcurve Counts per Year:")
    print(lengths_sem_table)  # NEU

    # ►► Funktion anpassen: SEM zusätzlich aufnehmen
    def save_table_as_csv(mean_table, std_table, sem_table, filename):
        combined_table = mean_table.copy()
        for col in mean_table.columns:
            combined_table[col] = (
                mean_table[col].astype(str)
                + " ± " + std_table[col].astype(str)
                + " (±" + sem_table[col].astype(str) + ")"   # NEU
            )
        combined_table.to_csv(HCorLC + "Lightcurves/" + filename, index=True)
        print(f"Table saved as {filename}")
    
    def save_table_as_csv_nosem(mean_table, std_table, filename):
        combined_table = mean_table.copy()
        for col in mean_table.columns:
            combined_table[col] = (
                mean_table[col].astype(str)
                + " ± " + std_table[col].astype(str)
                + ")"   # NEU
            )
        combined_table.to_csv(HCorLC + "Lightcurves/" + filename, index=True)
        print(f"Table saved as {filename}")

    save_table_as_csv(
        lengths_mean_table,
        lengths_std_table,
        lengths_sem_table,                       # NEU
        HCorLC + "rates_table.csv"
    )
    print("The rates tables have been saved.")

    # ---------- PDF-Tabelle --------------------------------------------------------
    with PdfPages(HCorLC + "Lightcurves/" + HCorLC + "rates_table.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')

        # ►► Tabelle: SD und SEM ausgeben
        table_data = [[
            f"{lengths_mean_table.loc[i, j]} ± {lengths_std_table.loc[i, j]} "
            f"(±{lengths_sem_table.loc[i, j]})"               # NEU
            for j in lengths_mean_table.columns
        ] for i in lengths_mean_table.index]

        plt.table(
            cellText=table_data,
            rowLabels=lengths_mean_table.index,
            colLabels=lengths_mean_table.columns,
            loc='center',
            cellLoc='center'
        )

        ax.set_title("Mean observation rate by σ threshold and minimal detections", pad=20)
        plt.figtext(0.5, 0.6, "detections", ha="center", fontsize=10)
        plt.figtext(0.08, 0.5, "sigma", va="center", rotation="vertical", fontsize=10)
        plt.figtext(
            0.5, 0.7,
            "How many visible SNeIa per year?  "
            "Value ± SD (±SEM)",                       # NEU
            ha="center",
            fontsize=10
        )
        pdf.savefig(fig)
        plt.close()
    print("The rates table pdf has been created and saved.")

    # ---------- Plot (bleibt unverändert – nur SD als y-Fehler) --------------------
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 18})
    colors = ['red', 'blue', 'green', 'purple', 'orange', "pink", "brown"]

    for idx, row in lengths_mean_table.iterrows():
        mean_values = row.astype(float)
        std_values  = lengths_std_table.loc[idx].astype(float)
        plt.errorbar(
            lengths_mean_table.columns.astype(int),
            mean_values,
            yerr=std_values,
            label=f"{idx}-σ threshold",
            capsize=5,
            fmt='-o',
            color=colors[idx - 1]
        )

    plt.xlabel('Minimal number of data points over σ threshold')
    plt.ylabel('Number of SNIa observations per year')
    plt.ylim(0,None)
    plt.ylim(0, 2200)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.xticks(lengths_mean_table.columns.astype(int), lengths_mean_table.columns)
    plt.tight_layout()
    plt.savefig(HCorLC + "Lightcurves/" + HCorLC + "rates_plot.pdf")
    print("The rates plot has been created and saved.")


#_________________________________    
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 18})
    colors = ['red', 'blue', 'green', 'purple', 'orange', "pink", "brown"]

    for idx, row in lengths_mean_table.iterrows():
        mean_values = row.astype(float)
        std_values  = lengths_std_table.loc[idx].astype(float)
        plt.plot(  # Errorbars rausgenommen
            lengths_mean_table.columns.astype(int),
            mean_values,
            '-o',
            label=f"{idx}-σ threshold",
            color=colors[idx - 1]
        )

    plt.xlabel('Minimal number of data points over σ threshold')
    plt.ylabel('Number of SNIa observations per year')

    # Y-Achse logarithmisch festlegen
    plt.yscale("log")
    plt.ylim(35, 6000)   # statt 0 ein kleiner Wert

    plt.legend(loc="upper right")
    plt.grid(alpha=0.3, which="both")  # both = major+minor log grid
    plt.xticks(lengths_mean_table.columns.astype(int), lengths_mean_table.columns)
    plt.tight_layout()
    plt.savefig(HCorLC + "Lightcurves/" + HCorLC + "rates_plot_log.pdf")
    print("The logarithmic rates plot has been created and saved.")




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
    save_table_as_csv_nosem(formatted_table_c_mean, formatted_table_c_std, "redshiftbin_c_table.csv")

    # Save the observation rates tables
    save_table_as_csv_nosem(formatted_table_n_mean, formatted_table_n_std, "redshiftbin_n_table.csv")

    print("CSV files saved successfully.")

    #____________________________________________________________________________________________________________________-
    # NEW: Colour binning with thresholds (3,3) within Redshift bins
    print("Starting colour binning analysis with thresholds...")

    # Define new Redshift bins (size 0.03 from 0 to 0.12)
    new_redshift_bins = [0, 0.03, 0.06, 0.09, 0.12]

    # Define Colour bins: 5 equally sized bins from -0.35 to 0.15
    color_bins = [-0.25,-0.225,-0.2,-0.175, -0.15,-0.125, -0.1, -0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35]

    # Optional: Create labels for the Colour bins (can be used for later analysis)
    color_bin_labels = [f"{color_bins[i]:.2f} to {color_bins[i+1]:.2f}" for i in range(len(color_bins) - 1)]

    # Dictionary to store normalized counts for the combined plot
    normalized_counts_dict = {}

    # Iterate over each Redshift bin and create separate plots
    for i in range(len(new_redshift_bins) - 1):
        # Initialize a counter for the number of lightcurves in each Colour bin
        counts = np.zeros(len(color_bins) - 1)
        
        # Loop through all files
        for file in text_files:
            try:
                # Read the dataset from the parquet file
                dataset = skysurvey.DataSet.read_parquet(file)
                dataset_df = dataset.data.copy()
                
                # Filter the data for the current Redshift bin
                mask = (dataset_df['z'] >= new_redshift_bins[i]) & (dataset_df['z'] < new_redshift_bins[i+1])
                redshift_subset = dataset_df[mask]
                
                # If there is no data in the current Redshift bin, skip the file
                if redshift_subset.empty:
                    continue
                
                # Set the data to the filtered subset
                dataset.set_data(redshift_subset)
                
                # Process lightcurve data with thresholds: min_sn_points=3 and min_detections=3
                lightcurves = process_lightcurve_data_rates(dataset, min_sn_points=3, min_detections=3)
                
                # If the processed lightcurve data is empty, skip the file
                if lightcurves.data.empty:
                    continue
                
                # Bin the 'c' values from the processed lightcurve data into the defined Colour bins
                color_indices = pd.cut(lightcurves.data["c"], bins=color_bins, labels=False, include_lowest=True)
                
                # Count the number of lightcurves in each Colour bin (normalize per year by dividing by num_years)
                for bin_idx in range(len(color_bins) - 1):
                    unique_lightcurves = lightcurves.data[color_indices == bin_idx].index.get_level_values(0).unique()
                    counts[bin_idx] += len(unique_lightcurves) / num_years
            except Exception as e:
                print(f"Error processing file {file} in colour binning analysis with thresholds: {e}")
        
        # Save the normalized counts for the combined plot later.
        # Normalize counts so that the sum for the current redshift bin equals 1 (if total > 0)
        total_counts = counts.sum()
        total_counts=1
        if total_counts > 0:
            normalized_counts = counts / total_counts
        else:
            normalized_counts = counts  # remains zero if no counts exist
        redshift_label = f"{new_redshift_bins[i]} to {new_redshift_bins[i+1]}"
        normalized_counts_dict[redshift_label] = normalized_counts
        
        # Create a stair plot for the current Redshift bin (individual plot)
        plt.figure(figsize=(7, 7))
        plt.rcParams.update({'font.size': 18})
        plt.stairs(
            values=counts,
            edges=color_bins,
            fill=False,
            linewidth=3,
            color="green",
            label=f"$z \\in [{new_redshift_bins[i]}, {new_redshift_bins[i+1]}]$"
        )
        #plt.xlim(-0.45, 0.21)
        plt.xlabel('Color Parameter Bins')
        plt.ylabel('Number of SNIa observations per year')
        plt.legend(loc="upper right")
        # plt.title(f"Redshift Bin: {redshift_label}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save the individual plot as a PDF file
        plt.savefig(HCorLC + "Lightcurves/" + f"redshiftbin_{new_redshift_bins[i]}_{new_redshift_bins[i+1]}_colour_count_plot_thresholds.pdf")
        plt.close()

    print("All individual Colour binning plots with thresholds have been created and saved.")

    # Create a combined normalized plot where counts for each redshift bin sum to 1
    print("Creating combined normalized plot for Colour binning with thresholds...")
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 18})
    # Define colors for each redshift bin (4 bins)
    colors = ['tab:blue', 'tab:green', 'tab:olive','tab:red' ]

    for idx, (redshift_label, norm_counts) in enumerate(normalized_counts_dict.items()):
        plt.stairs(
            values=norm_counts,
            edges=color_bins,
            fill=True,
            alpha=0.1,
            linewidth=3,
            color=colors[idx]
        )

    for idx, (redshift_label, norm_counts) in enumerate(normalized_counts_dict.items()):
        plt.stairs(
            values=norm_counts,
            edges=color_bins,
            fill=False,
            linewidth=3,
            color=colors[idx],
            label=f"$z \\in [{new_redshift_bins[idx]}, {new_redshift_bins[idx+1]}]$"
        )



    #plt.ylim(0,340)
    plt.xlabel('Color Parameter Bins')
    plt.ylabel('Number of SNIa observations per year')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the combined plot as a PDF file
    plt.savefig(HCorLC + "Lightcurves/" + "combined_normalized_colour_count_plot_thresholds.pdf")
    plt.close()

    print("The combined normalized Colour binning plot has been created and saved.")











    



if __name__ == "__main__":
    main()