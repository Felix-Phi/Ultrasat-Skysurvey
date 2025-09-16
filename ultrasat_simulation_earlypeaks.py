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

from templates import create_salt3_template
from lightcurves import process_lightcurve_data_earlypeaks
from survey_plan import zp_func_to_radoffset
from plotting import plot_survey_overview


def main():
    print("main() function is starting...")

    HCorLC = "HC_"
    folder_path = HCorLC + "Lightcurves/lightcurves"
    text_files = glob.glob(os.path.join(folder_path, "*.parquet"))
    num_years = len(text_files)
    if num_years == 0:
        print("No parquet files found.")
        return

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # SALT3 Model laden
    try:
        create_salt3_template(config["template_directory"])
        print("SALT3 template successfully created and registered!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # --------------------------------------------------------------------------
    # Vorverarbeitete Lightcurves pro Jahr einlesen
    preprocessed_lightcurves = {}
    for file in text_files:
        try:
            dataset = skysurvey.DataSet.read_parquet(file)
            lightcurves = process_lightcurve_data_earlypeaks(
                dataset, min_sn_points=1, min_detections=1
            )
            preprocessed_lightcurves[file] = lightcurves.data
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    print(preprocessed_lightcurves)
    # Parameter
    min_sn_points_values          = [1, 2, 3, 4, 5]
    max_time_after_explosion_vals = [1, 2, 3, 4, 5]

    # ►► Tabellen für Mean, SD und SEM
    rates_mean_table = pd.DataFrame(index=min_sn_points_values,
                                    columns=max_time_after_explosion_vals)
    rates_std_table  = pd.DataFrame(index=min_sn_points_values,
                                    columns=max_time_after_explosion_vals)
    rates_sem_table  = pd.DataFrame(index=min_sn_points_values,
                                    columns=max_time_after_explosion_vals)     # NEU

    # --------------------------------------------------------------------------
    for min_sn_points in min_sn_points_values:
        for max_time in max_time_after_explosion_vals:
            rates_per_file = []

            for file, lcv_df in preprocessed_lightcurves.items():
                # Filter: S/N > threshold & time < max_time
                sn_early = lcv_df[
                    (lcv_df["flux"] / lcv_df["fluxerr"] > min_sn_points) &
                    (lcv_df["time_after_explosion"] < max_time)
                ]
                valid_ids = sn_early.index.get_level_values(0).unique()

                # zusätzl. Bedingung: flux < (max_flux/3)
                max_flux = lcv_df.groupby(level=0)["flux"].max()
                selected = [
                    idx for idx in valid_ids
                    if (sn_early.loc[idx]["flux"] < max_flux[idx] / 3).any()
                ]

                rates_per_file.append(len(selected))

            # ►► Mean, SD, SEM
            if rates_per_file:
                mean_rate = round(np.mean(rates_per_file), 1)
                std_rate  = round(np.std(rates_per_file, ddof=0), 1)
                sem_rate  = round(std_rate / np.sqrt(len(rates_per_file)), 1)   # NEU
            else:
                mean_rate = std_rate = sem_rate = 0

            rates_mean_table.loc[min_sn_points, max_time] = mean_rate
            rates_std_table.loc[min_sn_points,  max_time] = std_rate
            rates_sem_table.loc[min_sn_points,  max_time] = sem_rate            # NEU

            print(f"Processed min_sn_points={min_sn_points}, max_time={max_time}")

    # --------------------------------------------------------------------------
    print("Mean table:")
    print(rates_mean_table)
    print("\nSD table:")
    print(rates_std_table)
    print("\nSEM table:")
    print(rates_sem_table)                                                       # NEU

    # --------------------------------------------------------------------------
    def save_table_as_csv(mean_table, std_table, filename, sem_table=None):
        """
        Speichert eine Tabelle als CSV.
        - Wenn sem_table übergeben wird:  Mean ± SD (±SEM)
        - Sonst:                           Mean ± SD
        """
        combined = mean_table.copy()
        for col in mean_table.columns:
            if sem_table is None:
                combined[col] = (mean_table[col].astype(str)
                                  + " ± " + std_table[col].astype(str))
            else:
                combined[col] = (mean_table[col].astype(str)
                                  + " ± " + std_table[col].astype(str)
                                  + " (±" + sem_table[col].astype(str) + ")")
        combined.to_csv(HCorLC + "Lightcurves/" + filename, index=True)
        print(f"Table saved as {filename}")

    # Rates-CSV speichern  (Mean ± SD ± SEM)
    save_table_as_csv(rates_mean_table, rates_std_table,
                      HCorLC + "early_rates_table.csv",
                      rates_sem_table)                                           # NEU

    # --------------------------------------------------------------------------
    # PDF-Tabelle mit SEM
    with PdfPages(HCorLC + "Lightcurves/" + HCorLC + "early_peaks_table.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')

        table_data = [[
            f"{rates_mean_table.loc[i, j]} ± {rates_std_table.loc[i, j]} "
            f"(±{rates_sem_table.loc[i, j]})"                                    # NEU
            for j in rates_mean_table.columns
        ] for i in rates_mean_table.index]

        plt.table(
            cellText=table_data,
            rowLabels=rates_mean_table.index,
            colLabels=rates_mean_table.columns,
            loc='center',
            cellLoc='center'
        )

        ax.set_title("Mean early SNIa observations per year "
                     "by σ threshold & max days", pad=20)
        plt.figtext(0.5, 0.6, "max days after explosion", ha="center", fontsize=10)
        plt.figtext(0.08, 0.5, "σ threshold", va="center",
                    rotation="vertical", fontsize=10)
        plt.figtext(0.5, 0.7,
                    "Value ± SD (±SEM)", ha="center", fontsize=10)               # NEU
        pdf.savefig(fig)
        plt.close()
    print("PDF table saved.")

    # --------------------------------------------------------------------------
    # Plots (weiterhin mit SD-Fehlerbalken)
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 18})
    colors = ['red', 'blue', 'green', 'purple', 'orange', "pink", "brown"]

    for idx, row in rates_mean_table.iterrows():
        plt.errorbar(
            rates_mean_table.columns.astype(int),
            row.astype(float),
            yerr=rates_std_table.loc[idx].astype(float),
            label=f"{idx}-σ threshold",
            capsize=5, fmt='-o', color=colors[idx - 1]
        )

    plt.xlabel('Max days after explosion')
    plt.ylabel('Early SNIa observations per year')
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(HCorLC + "Lightcurves/" + HCorLC + "early_peaks_plot.pdf")
    plt.close()
    print("Plot saved.")
    
    # Plot für die Raten mit Fehlerbalken (Error Bars)
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 18})
    colors = ['red', 'blue', 'green', 'purple', 'orange', "pink", "brown"]

    for idx, row in list(rates_mean_table.iterrows())[1:]:
        # Umwandeln in numerische Werte
        mean_values = row.astype(float)
        std_values = rates_std_table.loc[idx].astype(float)
        plt.errorbar(
            rates_mean_table.columns.astype(int),
            mean_values,
            yerr=std_values,
            label=f"{idx}-σ threshold",
            capsize=5,
            fmt='-o',
            color=colors[idx - 1]
        )

    plt.xlabel('Maximal days after explosion')
    plt.ylabel('Number of early SNIa observations per year')
    plt.ylim(0, 65)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.xticks(rates_mean_table.columns.astype(int), rates_mean_table.columns)
    plt.tight_layout()
    plt.savefig(HCorLC+"Lightcurves/"+HCorLC+"early_peaks_plot_focused.pdf")
    print("The early peaks focused plot has been created and saved.")

if __name__ == "__main__":
    main()
