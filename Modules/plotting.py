import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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

    
    control_points = [
    (0.0,    "#3b4cc0"),
    (0.125, "#f7f7f7"),
    (0.25, "#b40426"),
    (1.0,    "#300000"),
    ]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    "coolwarm_extended",
    control_points
    )
    norm = mcolors.Normalize(vmin=-0.15, vmax=1.05, clip=True)

    # Create the scatter plot
    plt.figure(figsize=(8, 7))
    plt.rcParams.update({'font.size': 18})
    plt.gca().invert_yaxis()
    print(data.head(10))
    # Scatter plot with color-coded reddening (c)
    scatter = plt.scatter(
        data['z'],                     # X-axis: Redshift
        data['USAT mag'],           # Y-axis: Minimum observed magnitude
        c=data['c'],                   # Color scale: Reddening
        cmap=custom_cmap, norm=norm,
        edgecolor='black',linewidths=0.3,
        s=60,                       # Point size
        label=("SN Ia with ≥"+str(min_detections)+" data points\nsurpassing S/N>"+str(min_sn_points))
    )

    # Add a colorbar for the reddening scale
    cbar = plt.colorbar(scatter)
    cbar.set_label('Color parameter c')

    # Annotate plot with the number of detectable SNIas
    plt.plot([], [], " ", label=f"# of SNe Ia above S/N = {len(data)}")

    # Add a horizontal line for the limiting magnitude
    #plt.axhline(y=22.5, linestyle="-", color="green", label="Limiting magnitude 22.5 mag")
    plt.rcParams.update({'font.size': 15})
    # Set plot labels and title
    plt.xlabel('Redshift z')
    plt.ylabel('ULTRASAT magnitude of brightest data point')
    if Alternative_Survey and hilocadence=="Low Cadence":
        #plt.title("1 day "+hilocadence+" Survey Simulation Over "+str(duration)+" days. Option 2.")
        plt.title("Option 2 1-day "+hilocadence+" Simulation.")
    elif hilocadence=="Low Cadence":
        #plt.title(str(cadence)+" days "+hilocadence+" Survey Simulation Over "+str(duration)+" days.")
        plt.title("Option 1 "+str(cadence)+"-day "+hilocadence+" Simulation.")
    elif hilocadence=="High Cadence":
        #plt.title(hilocadence+" Survey Simulation Over "+str(duration)+" days.")
        plt.title("15-min "+hilocadence+" Simulation.")
    else:
        raise "Something went wrong in the plotting."

    # Adjust plot limits
    plt.xlim(0, 0.132)
    plt.ylim(26, 16)

    # Add legend
    plt.legend(fontsize=15,loc="upper right")

    # Save and display the plot
    output_file = generate_unique_filename(folder,output_file)
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    if plot_show==True:
        plt.show()

def plot_survey_overview_mw(data, min_sn_points,min_detections,hilocadence,duration,observation_hours, cadence, Alternative_Survey, output_file,plot_show,folder="Results"):
    """
    Generates and saves a plot to visualize supernova observations.

    Args:
        data (pandas.DataFrame): Processed supernova data for plotting.
        indices (list): List of valid supernova indices.
        output_file (str): File name for saving the plot.
    """

    
    control_points = [
    (0.0, "#f7f7f7"),
    (0.25, "#b40426"),
    (1.0,    "#300000"),
    ]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    "coolwarm_extended",
    control_points
    )
    norm = mcolors.Normalize(vmin=0, vmax=5, clip=True)

    # Create the scatter plot
    plt.figure(figsize=(8, 7))
    plt.rcParams.update({'font.size': 18})
    plt.gca().invert_yaxis()
    print(data.head(10))
    # Scatter plot with color-coded milky way extinction (mwebv)
    scatter = plt.scatter(
        data['z'],                     # X-axis: Redshift
        data['USAT mag'],           # Y-axis: Minimum observed magnitude
        c=data['mwebv'],                   # Color scale: Milky Way extinction
        cmap=custom_cmap, norm=norm,
        edgecolor='black',linewidths=0.3,
        s=60,                       # Point size
        label=("SN Ia with ≥"+str(min_detections)+" data points\nsurpassing S/N>"+str(min_sn_points))
    )

    # Add a colorbar for the reddening scale
    cbar = plt.colorbar(scatter)
    cbar.set_label(r"Milky Way extinction E(B-V)$_{\mathrm{MW}}$")

    # Annotate plot with the number of detectable SNIas
    plt.plot([], [], " ", label=f"# of SNe Ia above S/N = {len(data)}")

    # Add a horizontal line for the limiting magnitude
    #plt.axhline(y=22.5, linestyle="-", color="green", label="Limiting magnitude 22.5 mag")
    plt.rcParams.update({'font.size': 15})
    # Set plot labels and title
    plt.xlabel('Redshift z')
    plt.ylabel('ULTRASAT magnitude of brightest data point')
    if Alternative_Survey and hilocadence=="Low Cadence":
        #plt.title("1 day "+hilocadence+" Survey Simulation Over "+str(duration)+" days. Option 2.")
        plt.title("Option 2 1-day "+hilocadence+" Simulation.")
    elif hilocadence=="Low Cadence":
        #plt.title(str(cadence)+" days "+hilocadence+" Survey Simulation Over "+str(duration)+" days.")
        plt.title("Option 1 "+str(cadence)+"-day "+hilocadence+" Simulation.")
    elif hilocadence=="High Cadence":
        #plt.title(hilocadence+" Survey Simulation Over "+str(duration)+" days.")
        plt.title("15-min "+hilocadence+" Simulation.")
    else:
        raise "Something went wrong in the plotting."

    # Adjust plot limits
    plt.xlim(0, 0.132)
    plt.ylim(26, 16)

    # Add legend
    plt.legend(fontsize=15,loc="upper right")

    # Save and display the plot
    output_file = (generate_unique_filename(folder,output_file))
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    if plot_show==True:
        plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_survey_overview_combined(data, *,
                                  min_sn_points,
                                  min_detections,
                                  hilocadence,
                                  duration,
                                  observation_hours,
                                  cadence,
                                  Alternative_Survey,
                                  output_file,
                                  plot_show,
                                  folder="Results"):
    """
    Combined overview plot:  
      • face colour  → SALT‑II colour parameter c  
      • rim colour   → Milky‑Way extinction E(B‑V)_MW

    Parameters
    ----------
    data : pandas.DataFrame
        Must contain columns  'z', 'USAT mag', 'c', 'mwebv'.
    …   (all other arguments identical to the original two routines)
    """

    # --- colormap for SN colour parameter c (same as before) ------------------
    face_control_points = [
        (0.00,  "#3b4cc0"),   # blue
        (0.125, "#f7f7f7"),   # almost white
        (0.25,  "#b40426"),   # red
        (1.00,  "#300000")    # very dark red
    ]
    face_cmap = mcolors.LinearSegmentedColormap.from_list(
        "c_cmap", face_control_points
    )
    face_norm = mcolors.Normalize(vmin=-0.15, vmax=1.05, clip=True)

    # --- colormap for MW reddening E(B‑V) (pale‑grey → black) -------------------
    edge_cmap = mcolors.LinearSegmentedColormap.from_list(
        "edge_cmap", ["#000000", "#FF9100"]   # start light grey, end black
    )
    edge_norm = mcolors.Normalize(vmin=0, vmax=1, clip=True)

    #edge_rgba  = edge_cmap(edge_norm(np.clip(data['mwebv'].values, 0, 2)))

    fig, ax = plt.subplots(figsize=(8, 7))        # +1 inch width for bars
    ax.invert_yaxis()
    plt.rcParams.update({'font.size': 18})

    edge_rgba = edge_cmap(edge_norm(np.clip(data['mwebv'].values, 0, 2)))

    sc = ax.scatter(data['z'],
                    data['USAT mag'],
                    c=data['c'],
                    cmap=face_cmap,
                    norm=face_norm,
                    edgecolors=edge_rgba,
                    linewidths=1.2,          # <‑‑ thicker rims
                    s=60,
                    label=("SN Ia with ≥"+str(min_detections)+" data points\nsurpassing S/N>"+str(min_sn_points))
                    )

    # ---------- colour‑bars in their own axes so they never touch --------------
    fig.subplots_adjust(right=0.80)               # leave 20 % of the width free

    # top bar – SN colour c
    cax_face = fig.add_axes([0.83, 0.55, 0.02, 0.30])   # [left, bottom, width, height]
    cbar_face = fig.colorbar(sc, cax=cax_face)
    cbar_face.set_label('Color $c$')

    # bottom bar – MW extinction
    cax_edge = fig.add_axes([0.83, 0.15, 0.02, 0.30])
    sm_edge  = plt.cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap)
    sm_edge.set_array([])
    cbar_edge = fig.colorbar(sm_edge, cax=cax_edge)
    cbar_edge.set_label(r"E(B-V)$_{\mathrm{MW}}$")

    # ----------------- axis labels & title -----------------------------------
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('ULTRASAT magnitude of brightest data point')

    if Alternative_Survey and hilocadence == "Low Cadence":
        title = "Option 2 1‑day Low‑Cadence Simulation"
    elif hilocadence == "Low Cadence":
        title = f"Option 1 {cadence}-day Low‑Cadence Simulation"
    elif hilocadence == "High Cadence":
        title = "15‑min High‑Cadence Simulation"
    else:
        raise ValueError("Unexpected cadence specification")

    ax.set_title(title, fontsize=17)

    # limits, legend, annotation
    ax.set_xlim(0, 0.132)
    ax.set_ylim(26, 16)
    
    ax.plot([], [], " ", label=f"# of SNe Ia above S/N = {len(data)}")  # dummy for legend
    ax.legend(fontsize=14, loc="upper right")

    # ------------- save / show ------------------------------------------------
    output_path = generate_unique_filename(folder, output_file)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    if plot_show:
        plt.show()
    plt.close(fig)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

def plot_survey_overview_with_mwlines(data,
                                      min_sn_points, min_detections,
                                      hilocadence, duration,
                                      observation_hours, cadence,
                                      Alternative_Survey,
                                      output_file, plot_show,
                                      folder="Results"):

    # ── 1. colormaps ───────────────────────────────────────────────
    # a) colour of the points: SN colour‑parameter c (diverging map)
    cmap_c   = mcolors.LinearSegmentedColormap.from_list(
        "cmap_c",
        [(0.00, "#3b4cc0"),
         (0.125,"#f7f7f7"),
         (0.25,"#b40426"),
         (1.00,"#300000")])
    norm_c   = mcolors.Normalize(vmin=-0.15, vmax=1.05, clip=True)

    # ── 2. figure ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.invert_yaxis()
    plt.rcParams.update({'font.size': 18})

    # ◆ 2a. scatter points coloured by c
    sc = ax.scatter(data['z'], data['USAT mag'],
                    c=data['c'], cmap=cmap_c, norm=norm_c,
                    edgecolor='black', linewidths=0.3, s=60,
                    zorder=3,
                    label=(f"SN Ia with ≥{min_detections} points\n"
                           f"S/N >{min_sn_points}")
                    )

    # ◆ 2b. vertical segments whose length = mwebv
    segments = [ [(z, mag), (z, mag + mw)]            # “downwards” segment
                 for z, mag, mw in
                 zip(data['z'], data['USAT mag'], data['mwebv']) ]
    lc = LineCollection(segments,color="black",
                        linewidths=1.8, zorder=2, alpha=0.9)
    lc.set_array(data['mwebv'])
    ax.add_collection(lc)

    # ── 3. axes, colour‑bars, legend ──────────────────────────────
    cbar_c  = fig.colorbar(sc,  ax=ax, label='Colour parameter c')

    plt.rcParams.update({'font.size': 15})
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('ULTRASAT magnitude of brightest data point')

    if Alternative_Survey and hilocadence == "Low Cadence":
        ax.set_title("Option 2 1‑day Low‑cadence Simulation")
    elif hilocadence == "Low Cadence":
        ax.set_title(f"Option 1 {cadence}-day Low‑cadence Simulation")
    elif hilocadence == "High Cadence":
        ax.set_title("15‑min High‑cadence Simulation")
    else:
        raise ValueError("Unexpected cadence setting")

    ax.set_xlim(0, 0.132)
    ax.set_ylim(26, 16)
    ax.legend(fontsize=15, loc="upper right")

    # ── 4. save / show ────────────────────────────────────────────
    output_file = generate_unique_filename(folder, output_file)
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    if plot_show:
        plt.show()
    plt.close(fig)

