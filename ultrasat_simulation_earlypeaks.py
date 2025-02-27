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
from lightcurves import process_lightcurve_data_rates
from survey_plan import zp_func_to_radoffset
from plotting import plot_survey_overview

def main():
    print("main() function is starting...")

    # Pfad zum Ordner mit den Parquet-Dateien
    HCorLC = "LClim_"
    folder_path = HCorLC + "Lightcurves/lightcurves"

    # Finde alle .parquet-Dateien im Ordner
    text_files = glob.glob(os.path.join(folder_path, "*.parquet"))
    num_years = len(text_files)
    if num_years == 0:
        print("No parquet files found.")
        return
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)


    # Create the SALT3 model
    try:
        create_salt3_template(config["template_directory"])
        print("SALT3 template successfully created and registered!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Vorverarbeitung: Alle Dateien einlesen und die vorverarbeiteten Lightcurves in einem Dictionary speichern
    preprocessed_lightcurves = {}
    for file in text_files:
        try:
            # Lese die Parquet-Datei ein
            dataset = skysurvey.DataSet.read_parquet(file)
            # Feste Einstellungen für die Vorverarbeitung (min_sn_points=1 und min_detections=1)
            lightcurves = process_lightcurve_data_rates(dataset, min_sn_points=1, min_detections=1)
            preprocessed_lightcurves[file] = lightcurves.data
            #print(lightcurves.data)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Parameter für die Auswertung
    min_sn_points_values = [1, 2, 3, 4, 5]
    max_time_after_explosion_values = [1, 2, 3, 4, 5]

    # Erstelle leere Tabellen für Mittelwerte und Standardabweichungen der Raten
    rates_mean_table = pd.DataFrame(index=min_sn_points_values, columns=max_time_after_explosion_values)
    rates_std_table = pd.DataFrame(index=min_sn_points_values, columns=max_time_after_explosion_values)

    # Für jede Parameterkombination werden die Rates pro Datei (Jahr) berechnet
    for min_sn_points in min_sn_points_values:
        for max_time in max_time_after_explosion_values:
            rates_per_file = []
            for file, lightcurves_df in preprocessed_lightcurves.items():
                # Hier kannst du weitere Filter auf den vorverarbeiteten DataFrame anwenden
                # Beispiel: Filtere basierend auf min_sn_points und max_time_after_explosion

                    # Filtere für frühzeitige Peaks: 
                    #  - Bedingung: S/N (flux/fluxerr) > min_sn_points
                    #  - und Phase < max_phase
                sn_earlypeaks = lightcurves_df[
                    (lightcurves_df["flux"] / lightcurves_df["fluxerr"] > min_sn_points) &
                    (lightcurves_df["time_after_explosion"] < max_time)
                ]
                valid_indices_sn_phase = sn_earlypeaks.index.get_level_values(0).unique()

                # Zusätzliche Bedingung: Zum Zeitpunkt der Phase < max_phase ist der Flux
                # mindestens dreimal kleiner als der maximale Flux des jeweiligen Objekts
                max_flux_per_index = lightcurves_df.groupby(level=0)["flux"].max()
                valid_indices_flux_check = []
                for idx in valid_indices_sn_phase:
                    max_flux = max_flux_per_index[idx]
                    sub_data = sn_earlypeaks[sn_earlypeaks.index.get_level_values(0) == idx]
                    if (sub_data["flux"] < max_flux / 3).any():
                        valid_indices_flux_check.append(idx)

                # Die Rate entspricht der Anzahl der gültigen Indices (Objekte) in diesem Jahr
                rate = len(valid_indices_flux_check)
                rates_per_file.append(rate)


            # Berechne Mittelwert und Standardabweichung über alle Dateien
            if rates_per_file:
                mean_rate = round(np.mean(rates_per_file), 1)
                std_rate = round(np.std(rates_per_file), 1)
            else:
                mean_rate = 0
                std_rate = 0

            rates_mean_table.loc[min_sn_points, max_time] = mean_rate
            rates_std_table.loc[min_sn_points, max_time] = std_rate
            print(f"Processed min_sn_points={min_sn_points}, max_phase={max_time}")

    # Ergebnisse ausgeben
    print("Mean Table of Early SNIa Observations per Year:")
    print(rates_mean_table)
    print("\nStandard Deviation Table of Early SNIa Observations per Year:")
    print(rates_std_table)

    # Funktion, um die Tabellen als CSV zu speichern (mit Mean ± Std)
    def save_table_as_csv(mean_table, std_table):
        combined_table = mean_table.copy()
        for col in mean_table.columns:
            combined_table[col] = mean_table[col].astype(str) + " ± " + std_table[col].astype(str)
        combined_table.to_csv(HCorLC+"Lightcurves/"+ HCorLC + "early_rates_table.csv", index=True)
        print(f"Table saved as {HCorLC+"Lightcurves/"+ HCorLC + "early_rates_table.csv"}")

    save_table_as_csv(rates_mean_table, rates_std_table)

    # Speichern der Tabelle als PDF
    with PdfPages(HCorLC+"Lightcurves/"+HCorLC+"early_peaks_table.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')

        # Formatiere die Tabelleneinträge als "Mean ± Std"
        table_data = [[f"{rates_mean_table.loc[i, j]} ± {rates_std_table.loc[i, j]}"
                       for j in rates_mean_table.columns] for i in rates_mean_table.index]

        plt.table(
            cellText=table_data,
            rowLabels=rates_mean_table.index,
            colLabels=rates_mean_table.columns,
            loc='center',
            cellLoc='center'
        )

        ax.set_title("Mean Early SNIa Observations per Year\nby σ threshold and max phase", pad=20)
        plt.figtext(0.5, 0.6, "max time after explosion (days)", ha="center", fontsize=10)
        plt.figtext(0.08, 0.5, "σ threshold", va="center", rotation="vertical", fontsize=10)
        plt.figtext(0.5, 0.7, "Number of early SNIa observations per year", ha="center", fontsize=10)

        pdf.savefig(fig)
        plt.close()
    print("The early rates table pdf has been created and saved.")

    # Plot für die Raten mit Fehlerbalken (Error Bars)
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 18})
    colors = ['red', 'blue', 'green', 'purple', 'orange', "pink", "brown"]

    for idx, row in rates_mean_table.iterrows():
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
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.xticks(rates_mean_table.columns.astype(int), rates_mean_table.columns)
    plt.tight_layout()
    plt.savefig(HCorLC+"Lightcurves/"+HCorLC+"early_peaks_plot.pdf")
    print("The early peaks plot has been created and saved.")

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
    plt.ylim(0, 90)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.xticks(rates_mean_table.columns.astype(int), rates_mean_table.columns)
    plt.tight_layout()
    plt.savefig(HCorLC+"Lightcurves/"+HCorLC+"early_peaks_plot_focused.pdf")
    print("The early peaks focused plot has been created and saved.")

    # create plot for rates focused
    #plt.figure(figsize=(10, 6))
    #plt.rcParams.update({'font.size': 15})
    #colors = ['red', 'blue', 'green', 'purple', 'orange',"pink","brown"]  # Farben für die Linien
    #for idx, row in list(lengths_table.iterrows())[1:]:
    #    plt.plot(lengths_table.columns,row.values.astype(float),  alpha=1, color=colors[idx-1], label=str(idx)+"-σ threshold", marker='o')
    #plt.xlabel('maximal phase')
    #plt.ylabel('Number of yearly observations')
    #plt.title('Trend of the Visible Early Peaks regarding the σ thresholding')
    #plt.ylim(-3,20)
    #plt.legend()
    #plt.grid(alpha=0.3)
    #plt.xticks(lengths_table.columns, lengths_table.columns)
    #plt.tight_layout()
    #plt.savefig(HCorLC+"Lightcurves/"+HCorLC+"early_peaks_plot_focused.pdf")

    #plot the lightcurves
    #for i in lightcurves_df.index.get_level_values(0).unique():
    
    #print("Plotting lightcurves...")
    #for i in valid_indices_flux_check[:50]:
    #    df_index = sn_earlypeaks.loc[i]
    #    # Create the scatter plot
    #    plt.figure(figsize=(7, 7))
    #    plt.rcParams.update({'font.size': 18})
    #    plt.errorbar(df_index['phase'], df_index['mag'], yerr=df_index['magerr'],
    #                ecolor="gray", fmt='none', ls="None", elinewidth=2, alpha=1, zorder=1)
    #    plt.scatter(df_index['phase'], df_index['mag'], s=30, color="tab:blue",
    #                label="Simulated light curve", zorder=2)
    #    
    #    # Achsen und Titel
    #    plt.gca().invert_yaxis() 
    #    plt.xlim(-22, 20)
    #    plt.ylim(26, 16)
    #    
    #    # Weitere Informationen in der Legende
    #    plt.plot([], [], " ", label=("z = " + str(sn_earlypeaks.loc[i].iloc[0]["z"])))
    #    plt.plot([], [], " ", label=("c = " + str(sn_earlypeaks.loc[i].iloc[0]["c"])))
    #    
    #    # Alle einzigartigen zp-Werte aus der Gruppe abrufen
    #    zp_values = sn_earlypeaks.loc[i]["zp"].unique()
    #    
    #    # Prüfen, ob es mehr als einen Wert gibt
    #    if len(zp_values) > 1:
    #        for idx, zp_val in enumerate(zp_values, start=1):
    #            radial_offset = zp_func_to_radoffset(zp_val)
    #            plt.plot([], [], " ", label=f"radial offset #{idx} = {round(radial_offset, 2)}")
    #    else:
    #        radial_offset = zp_func_to_radoffset(zp_values[0])
    #        plt.plot([], [], " ", label=f"radial offset = {round(radial_offset, 2)}")
    #    
    #    plt.xlabel('Phase in days')
    #    plt.ylabel('Brightness in mag')
    #    plt.legend(loc="best", fontsize=18)
    #    plt.savefig(HCorLC+"Lightcurves/plots_lightcurves/survey_example_" + str(i) + ".png",
    #                dpi=600, bbox_inches='tight')


            
            







    



if __name__ == "__main__":
    main()