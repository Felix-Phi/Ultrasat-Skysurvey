import yaml
import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import skysurvey
import seaborn as sns

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
    min_detections_values = [1,2, 3,4, 5]

    # Create an empty table to store the dataset lengths
    lengths_table = pd.DataFrame(index=min_sn_points_values, columns=min_detections_values)

    backup_lightcurves=combined_dataset.data.copy()

    # Populate the table with dataset lengths
    for min_sn_points in min_sn_points_values:
        for min_detections in min_detections_values:
            dataset.set_data(backup_lightcurves)
            lightcurves=process_lightcurve_data_rates(
                dataset, min_sn_points=min_sn_points, min_detections=min_detections
            )
            #get the rate and put it in the table.
            lengths_table.loc[min_sn_points, min_detections] = round(len(lightcurves.data.index.get_level_values(0).unique().tolist())/num_years,1)
            
            


            
            

    # Convert values to integers for better display
    print(lengths_table)

    # Save the table as a PDF
    with PdfPages(HCorLC+"Lightcurves/"+HCorLC+"rates_table.pdf") as pdf:
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
        ax.set_title("Mean observation rate by sigma threshold and minimal detections", pad=20)
        plt.figtext(0.5, 0.6, "detections", ha="center", fontsize=10)
        plt.figtext(0.08, 0.5, "sigma", va="center", rotation="vertical", fontsize=10)
        plt.figtext(0.5, 0.7, "How many visible SNeIa per year?", ha="center", fontsize=10)

        #plt.show()
        pdf.savefig(fig)
        plt.close()

    

    print("The table has been created and saved.")

    # create plot for rates
    plt.figure(figsize=(9, 7))
    plt.rcParams.update({'font.size': 15})
    colors = ['red', 'blue', 'green', 'purple', 'orange',"pink","brown"]  # Farben für die Linien
    print(lengths_table.columns)
    for idx, row in lengths_table.iterrows():
        plt.plot(lengths_table.columns,row.values.astype(float),  alpha=1, color=colors[idx-1], label=str(idx)+"σ thresholding", marker='o')
    plt.xlabel('minimal detections')
    plt.ylabel('Number of yearly observations')
    plt.title('Trend of the Observation Counts regarding the sigma thresholding')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(lengths_table.columns, lengths_table.columns)
    plt.tight_layout()
    plt.savefig(HCorLC+"Lightcurves/"+HCorLC+"rates_plot.pdf")



    #_________________________________________________________-
    #REDSHIFT BINS

    # Define bins
    bins = [0, 0.01, 0.02, 0.03, 0.04,0.05, 0.06,0.07, 0.08,0.09, 0.10,0.11]
    
    # Create bin labels that describe the range
    bin_labels = [
    f"{str(f'{bins[i]:.2f}')[2:]}_to_{str(f'{bins[i+1]:.2f}')[2:]}" for i in range(len(bins) - 1)
    ]

    # Create a new column for bins
    combined_df['z_bin'] = pd.cut(combined_df['z'], bins=bins, labels=bin_labels, right=False)

    # Group by bins and create separate DataFrames
    binned_dfs = {label: group.drop(columns='z_bin') for label, group in combined_df.groupby('z_bin')}

    # Parameters for the table
    min_sn_points_values = [1,2,3,4,5]

    # Create an empty table to store the dataset lengths
    formatted_table_c = pd.DataFrame(index=min_sn_points_values, columns=bin_labels)
    formatted_table_n = pd.DataFrame(index=min_sn_points_values, columns=bin_labels)

    # Populate the table with dataset lengths
    for min_sn_points in min_sn_points_values:
        for bin in bin_labels:
            dataset.set_data(binned_dfs[bin])
            lightcurves=process_lightcurve_data_rates(
                dataset, min_sn_points=min_sn_points, min_detections=1
            )
            #get the rate and put it in the table.
            formatted_table_c.loc[min_sn_points, bin] = round(float(lightcurves.data["c"].mean()),3)
            formatted_table_n.loc[min_sn_points, bin] = round(len(lightcurves.data.index.get_level_values(0).unique().tolist())/num_years,1)


    # create plot for reddening
    plt.figure(figsize=(9, 7))
    plt.rcParams.update({'font.size': 15})
    colors = ['red', 'blue', 'green', 'purple', 'orange',"pink","brown"]  # Farben für die Linien

    print(formatted_table_n)
    
    for idx, row in formatted_table_c.iterrows():
        plt.stairs(values=row.values.astype(float), edges=bins, alpha=1, color=colors[idx-1],fill=False, label=str(idx)+"σ thresholding",baseline=None)
    plt.xlabel('Redshift Bins')
    plt.ylabel('reddening c')
    plt.title('Trend of the reddening parameter c as a Function of Redshift')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(HCorLC+"Lightcurves/"+HCorLC+"redshiftbin_c_plot.pdf")

    # create plot for rates in redshiftbins
    plt.figure(figsize=(9, 7))
    plt.rcParams.update({'font.size': 15})
    colors = ['red', 'blue', 'green', 'purple', 'orange',"pink","brown"]  # Farben für die Linien

    print(formatted_table_n)
    bin_edges=np.arange(start=0,stop=0.12,step=0.01)
    print(bin_edges)
    for idx, row in formatted_table_n.iterrows():
        plt.stairs(values=row.values.astype(float), edges=bins, alpha=1, color=colors[idx-1],fill=False, label=str(idx)+"σ thresholding",baseline=None)
    plt.xlabel('Redshift Bins')
    plt.ylabel('Number of yearly observations')
    plt.title('Trend of the Observation Count as a Function of Redshift')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot anzeigen
    plt.tight_layout()
    plt.savefig(HCorLC+"Lightcurves/"+HCorLC+"redshiftbin_n_plot.pdf")




    # Save the table as a PDF
    with PdfPages(HCorLC+"Lightcurves/"+HCorLC+"redshiftbin_c_table.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size
        ax.axis('tight')
        ax.axis('off')

        # Create the table
        table = plt.table(
            cellText=formatted_table_c.values,
            rowLabels=formatted_table_c.index,
            colLabels=formatted_table_c.columns,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)  # Adjust table font size
        table.scale(1.2, 1.2)  # Adjust table size (width, height)

        # Add axis labels and title
        ax.set_title("Mean reddening by sigma clipping and redshift bin",pad=-25, fontsize=13)
        plt.figtext(0.5, 0.6, "redshift bins", ha="center", fontsize=10)
        plt.figtext(0, 0.5, "sigma", va="center", rotation="vertical", fontsize=10)

        #plt.show()
        pdf.savefig(fig)
        plt.close()
    
    # Save the table as a PDF
    with PdfPages(HCorLC+"Lightcurves/"+HCorLC+"redshiftbin_rates_table.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size
        ax.axis('tight')
        ax.axis('off')

        # Create the table
        table = plt.table(
            cellText=formatted_table_n.values,
            rowLabels=formatted_table_n.index,
            colLabels=formatted_table_n.columns,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)  # Adjust table font size
        table.scale(1.2, 1.2)  # Adjust table size (width, height)

        # Add axis labels and title
        ax.set_title("Mean observation rate by sigma clipping and redshift bin",pad=-25, fontsize=13)
        plt.figtext(0.5, 0.6, "redshift bins", ha="center", fontsize=10)
        plt.figtext(0, 0.5, "sigma", va="center", rotation="vertical", fontsize=10)

        #plt.show()
        pdf.savefig(fig)
        plt.close()







    



if __name__ == "__main__":
    main()