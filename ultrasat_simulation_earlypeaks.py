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
    # Path to the folder containing the text files
    HCorLC="HC_"
    folder_path = HCorLC+"Lightcurves/lightcurves"

    # Find all .txt files in the folder
    text_files = glob.glob(os.path.join(folder_path, "*.parquet"))

    # List to store DataFrames
    dataframes = []
    num_years=len(text_files)
    
    # Read each file and append to the list
    add=0
    for file in text_files:
        try:
            dataset = skysurvey.DataSet.read_parquet(file)
            dataset_df=dataset.data
            new_level_0 = dataset_df.index.get_level_values(0) + add
            add=add+max(dataset_df.index.get_level_values(0).unique().tolist())
            
            new_index = pd.MultiIndex.from_arrays(
                [new_level_0, dataset_df.index.get_level_values(1)],
                names=dataset_df.index.names
            )
            dataset_df.index=new_index
            
            dataframes.append(dataset_df)
            
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    # Combine all DataFrames into one
    if dataframes:
        combined_df = pd.concat(dataframes)
        print("Combined DataFrame:")
        print(combined_df.head())  # Display the first few rows
    else:
        print("No files found to process.")

    # Optional: Save the combined DataFrame
    output_path = os.path.join(folder_path, "combined/combined_lightcurves.parquet")
    combined_df.to_parquet(output_path)
    print("Anzahl der Indezes")
    print(len(combined_df.index.get_level_values(0).unique().tolist())/num_years)
    combined_dataset=dataset
    combined_dataset.set_data(combined_df)
    print(f"The combined data has been saved to {output_path}.")

    #TestingLC=process_lightcurve_data_rates(dataset=combined_dataset,min_detections=1,min_sn_points=1)
    #print(len(TestingLC.data.index.get_level_values(0).unique().tolist()))


    # Parameters for the table
    min_sn_points_values = [1,2, 3,4, 5]
    max_phase_values=[-15,-16,-17,-18,-19]

    # Create an empty table to store the dataset lengths
    lengths_table = pd.DataFrame(index=min_sn_points_values, columns=max_phase_values)
    backup_lightcurves=combined_dataset.data.copy()

    # Populate the table with dataset lengths
    for min_sn_points in min_sn_points_values:
        for max_phase in max_phase_values:
            dataset.set_data(backup_lightcurves)
            lightcurves=process_lightcurve_data_rates(
                dataset, min_sn_points=1, min_detections=1
            )
            #Identify indices with at least one S/N > 3 and phase < -15
            lightcurves_df=lightcurves.data
            sn_earlypeaks = lightcurves_df[(lightcurves_df["flux"] / lightcurves_df["fluxerr"] > min_sn_points) & (lightcurves_df["phase"] < max_phase)]
            valid_indices_sn_phase = sn_earlypeaks.index.get_level_values(0).unique()
            
            # Additional condition: Flux at phase < max_phase is at least x times smaller than max flux
            max_flux_per_index = lightcurves_df.groupby(level=0)["flux"].max()
            valid_indices_flux_check = []
            for idx in valid_indices_sn_phase:
                max_flux = max_flux_per_index[idx]
                sub_data = sn_earlypeaks[
                    sn_earlypeaks.index.get_level_values(0) == idx
                ]
                if (sub_data["flux"] < max_flux / 3).any():
                    valid_indices_flux_check.append(idx)

            # Keep only data for these valid indices
            lightcurves_df = lightcurves_df[
                lightcurves_df.index.get_level_values(0).isin(valid_indices_flux_check)
            ]
            lightcurves.set_data(lightcurves_df)
            #get the rate and put it in the table.
            lengths_table.loc[min_sn_points, max_phase] = round(len(lightcurves.data.index.get_level_values(0).unique().tolist())/num_years,1)
            
    # create plot for rates
    plt.figure(figsize=(9, 7))
    plt.rcParams.update({'font.size': 15})
    colors = ['red', 'blue', 'green', 'purple', 'orange',"pink","brown"]  # Farben für die Linien
    for idx, row in lengths_table.iterrows():
        plt.plot(lengths_table.columns,row.values.astype(float),  alpha=1, color=colors[idx-1], label=str(idx)+"σ thresholding", marker='o')
    plt.xlabel('maximal phase')
    plt.ylabel('Number of yearly observations')
    plt.title('Trend of the Counts of Visible Early Peaks regarding the sigma thresholding')
    #plt.ylim(-3,20)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(lengths_table.columns, lengths_table.columns)
    plt.tight_layout()
    plt.savefig(HCorLC+"Lightcurves/"+HCorLC+"early_peaks_plot.pdf")

    # create plot for rates focused
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 15})
    colors = ['red', 'blue', 'green', 'purple', 'orange',"pink","brown"]  # Farben für die Linien
    for idx, row in list(lengths_table.iterrows())[1:]:
        plt.plot(lengths_table.columns,row.values.astype(float),  alpha=1, color=colors[idx-1], label=str(idx)+"σ-clipped", marker='o')
    plt.xlabel('maximal phase')
    plt.ylabel('Number of yearly observations')
    plt.title('Trend of the Counts of Visible Early Peaks regarding the clipping')
    plt.ylim(-3,20)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(lengths_table.columns, lengths_table.columns)
    plt.tight_layout()
    plt.savefig(HCorLC+"Lightcurves/"+HCorLC+"early_peaks_plot_focused.pdf")

    #plot the lightcurves
    for i in lightcurves_df.index.get_level_values(0).unique():
        df_index=lightcurves_df.loc[i]
        plt.figure(figsize=(9, 7))
        plt.rcParams.update({'font.size': 15})
        plt.errorbar(df_index['phase'], df_index['mag'], yerr=df_index['magerr'],ecolor="gray", fmt='none', ls="None", elinewidth=1,alpha=1,zorder=1)
        plt.scatter(df_index['phase'], df_index['mag'],s=10,color="tab:blue",label="Observed Data",zorder=2)
        # Achsen und Titel
        plt.gca().invert_yaxis() 
        plt.xlim(-22,20)
        #plt.ylim(25,16)
        #plt.plot(np.linspace(-20, 20, 100),line_mag,color="tab:blue",label="full lightcurve by")
        #plt.plot([],[]," ",label=("improved Salt3"))
        #plt.axhline(y=22.5,linestyle="-",color="green",label="limiting magnitude 22.5mag")
        plt.plot([],[]," ",label=("z="+str(lightcurves_df.loc[i].iloc[0]["z"])))
        plt.plot([],[]," ",label=("c="+str(lightcurves_df.loc[i].iloc[0]["c"])))
        plt.xlabel('Phase (days)')
        plt.ylabel('Magnitude (mag)')
        plt.title("Example simulation lightcurve")
        plt.legend(loc="lower right",fontsize=15)
        plt.savefig(HCorLC+"Lightcurves/plots_lightcurves/survey_example_"+str(i)+".png", dpi=600, bbox_inches='tight')



            
            

    # Convert values to integers for better display

    print(lengths_table)

    # Save the table as a PDF
    with PdfPages(HCorLC+"Lightcurves/"+HCorLC+"early_peak_table.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')
        plt.table(
            cellText=lengths_table.values,
            rowLabels=lengths_table.index,
            colLabels=lengths_table.columns,
            loc='center',
            cellLoc='center'
        )

        # Add axis labels and title
        ax.set_title("Early peak rates by sigma clipping and maximal phase. max-min>3 in flux.", pad=-20)
        plt.figtext(0.5, 0.6, "phase", ha="center", fontsize=10)
        plt.figtext(0.08, 0.5, "sigma", va="center", rotation="vertical", fontsize=10)
        plt.figtext(0.5, 0.7, "How many visible early peaks per year?", ha="center", fontsize=10)

        #plt.show()
        pdf.savefig(fig)
        plt.close()

    print("The table has been created and saved as '"+HCorLC+"early_peak_table.pdf'.")





    



if __name__ == "__main__":
    main()