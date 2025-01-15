import numpy as np
import os
import matplotlib.pyplot as plt


def extract_data_for_plotting(lightcurves, sniainstance):
    """
    Extracts and processes data for plotting based on detection criteria.

    Args:
        dataset (skysurvey.DataSet): Filtered light curve dataset.
        sniadata (pandas.DataFrame): Original supernova data.
        min_detections (int): Minimum number of detectable observations per supernova.

    Returns:
        tuple: Processed DataFrame for plotting and indices of valid supernovae.
    """

    dataset=lightcurves.data
    # Extract unique indices of supernovae that meet the detection criteria
    valid_indices = []
    for i in np.arange(len(dataset)):
        if dataset.index[i][0] not in valid_indices:
            valid_indices.append(dataset.index[i][0])

    # Extract the minimum observed magnitude for each supernova
    highest_mag_list = dataset.groupby(lightcurves._data_index)["mag"].min().reset_index()

    # Update the "USAT mag" column in the supernova data
    sniainstance.data["USAT mag"] = 0
    sniainstance.data["USAT mag"] = sniainstance.data["USAT mag"].astype(float)
    for idx, mag in zip(highest_mag_list['index'], highest_mag_list['mag']):
        if idx in sniainstance.data.index:
            sniainstance.data.at[idx, 'USAT mag'] = mag

    # Filter the supernova data to include only valid indices
    filtered_data = sniainstance.data.iloc[valid_indices]

    return filtered_data, highest_mag_list



def generate_unique_filename(directory, base_filename):
    """
    Generates a unique filename in the specified directory. If the file already exists,
    appends a numeric suffix (e.g., _1, _2) to the filename.

    Args:
        directory (str): Directory where the file will be saved.
        base_filename (str): Desired base filename (e.g., "file.png").

    Returns:
        str: Unique filename with the path.
    """
    # Extract the file name and extension
    name, ext = os.path.splitext(base_filename)

    # Check if the file already exists
    unique_filename = os.path.join(directory, base_filename)
    counter = 1
    while os.path.exists(unique_filename):
        # Append a suffix to the filename
        unique_filename = os.path.join(directory, f"{name}_{counter}{ext}")
        counter += 1

    return unique_filename

def plot_survey_overview(data, min_sn_points,min_detections,hilocadence,duration,observation_hours, cadence, Alternative_Survey, output_file,plot_show,folder="Results"):
    """
    Generates and saves a plot to visualize supernova observations.

    Args:
        data (pandas.DataFrame): Processed supernova data for plotting.
        indices (list): List of valid supernova indices.
        output_file (str): File name for saving the plot.
    """
    # Create the scatter plot
    plt.figure(figsize=(9, 7))
    plt.rcParams.update({'font.size': 15})
    plt.gca().invert_yaxis()
    
    # Scatter plot with color-coded reddening (c)
    scatter = plt.scatter(
        data['z'],                     # X-axis: Redshift
        data['USAT mag'],           # Y-axis: Minimum observed magnitude
        c=data['c'],                   # Color scale: Reddening
        cmap='coolwarm',               # Colormap
        vmin=-0.15,
        vmax=0.15,
        edgecolor='black',
        s=100,                         # Point size
        label=("SNIa with ≤"+str(min_detections)+" detectable Obs. (S/N>"+str(min_sn_points)+")")
    )

    # Add a colorbar for the reddening scale
    cbar = plt.colorbar(scatter)
    cbar.set_label('Reddening c')

    # Annotate plot with the number of detectable SNIas
    plt.plot([], [], " ", label=f"# of detectable SNeIa = {len(data)}")

    # Add a horizontal line for the limiting magnitude
    #plt.axhline(y=22.5, linestyle="-", color="green", label="Limiting magnitude 22.5 mag")
    plt.rcParams.update({'font.size': 13})
    # Set plot labels and title
    plt.xlabel('Redshift z')
    plt.ylabel('Observed Ultrasat magnitude')
    if Alternative_Survey and hilocadence=="Low Cadence":
        plt.title("1 day "+hilocadence+" Survey Simulation Over "+str(duration)+" days. Option 2.")
    elif hilocadence=="Low Cadence":
        plt.title(str(cadence)+" days "+hilocadence+" Survey Simulation Over "+str(duration)+" days.")
    elif hilocadence=="High Cadence":
        plt.title(hilocadence+" Survey Simulation Over "+str(duration)+" days.")
    else:
        raise "Something went wrong in the plotting."

    # Adjust plot limits
    plt.xlim(0, 0.12)
    plt.ylim(25, 16)

    # Add legend
    plt.legend(fontsize=15)

    # Save and display the plot
    output_file = generate_unique_filename(folder,output_file)
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    if plot_show==True:
        plt.show()
