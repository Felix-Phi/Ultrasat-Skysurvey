import numpy as np
import math
from shapely import geometry
from skysurvey import Survey
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def create_footprint():
    """
    Creates the ULTRASAT footprint as a square polygon with an area of 204 square degrees.
    
    Returns:
        shapely.geometry.Polygon: ULTRASAT square footprint.
    """
    # Desired area in square degrees
    area = 204.0
    
    # Side length of a square with the given area
    side_length = math.sqrt(area)
    
    # Half the side length for centering the square at (0,0)
    half_side = side_length / 2.0
    
    # Create a square polygon centered at (0,0)
    # Coordinates: (−half_side, −half_side) to (half_side, half_side)
    footprint = geometry.box(-half_side, -half_side, half_side, half_side)
    
    return footprint

def create_footprint_circular():
    """
    Creates the ULTRASAT footprint as a circular buffer.
    Returns:
        shapely.geometry.Polygon: ULTRASAT footprint.
    """
    center_point = geometry.Point(0, 0)
    usat_fov = center_point.buffer(math.sqrt(170 / math.pi))  # Approximate area
    return geometry.Polygon(usat_fov)


def generate_time_array(start_day, end_day, time_step, observation_hours, pause_start_hour):
    """
    Generates a time array excluding specified pause hours.

    Args:
        start_day (float): Starting MJD.
        end_day (float): Ending MJD.
        time_step (float): Time step in Julian days.
        observation_hours (int): Number of observation hours per day.
        pause_start_hour (int): Start of pause period in hours.

    Returns:
        np.ndarray: Array of observation times (MJD).
    """
    days = np.arange(start_day, end_day, 1)
    daily_times = []

    for day in days:
        for quarter_hour in range(int(1/time_step)):  # quarter-hours in a day
            hour_of_day = quarter_hour *(time_step*24)
            if not (pause_start_hour <= hour_of_day < pause_start_hour + (24 - observation_hours)):
                daily_times.append(day + quarter_hour * time_step)

    print(len(daily_times))
    return np.array(daily_times)

def generate_field_coordinates_option1(start_day, end_day, time_step, observation_hours, pause_start_hour,cadence):
    """
    Generates RA and Dec coordinates for survey fields. Use the first option in the currently discussed approach.
    Args:
        start_day:
        end_day:
        timestep:
        observation hours:
        cadence:

    Returns:
        tuple: Arrays of Ra and Dec coordinates
    """

    # Pfad anpassen, falls sich die Datei in einem anderen Verzeichnis befindet
    df = pd.read_csv("Data/LCS_nonoverlapping_grid.csv", sep=",") 

    if cadence == 3:
        df = df[(df["V180"] == 1) & (df["A_U<1"] == 1)]

    elif cadence == 4:
        df = df[(df["V180"] == 1)]

    else:
        raise "Cadence needs to be either 3 (apply extinction limit) or 4 (without)"
    
    days = np.arange(start_day, end_day, 1)
    daily_times = []

    for day in days:
        for quarter_hour in range(int(1/time_step)):  # quarter-hours in a day
            hour_of_day = quarter_hour *(time_step*24)
            if not (pause_start_hour <= hour_of_day < pause_start_hour + (24 - observation_hours)):
                daily_times.append(day + quarter_hour * time_step)
    
    size = len(daily_times)

    df_first= df[((df["Dec"] > 0) | (df["Dec"] < -58))]
    df_second= df[((df["Dec"] < 0) | (df["Dec"] > 58))]
    ra_values_first = df_first["RA"].to_numpy()
    dec_values_first = df_first["Dec"].to_numpy()
    ra_values_second = df_second["RA"].to_numpy()
    dec_values_second = df_second["Dec"].to_numpy()

    LowCad_ra, LowCad_dec = [], []
    i_first, i_second = 0, 0                    # ← separate counters
    entries_per_day   = observation_hours / (time_step * 24)

    while len(LowCad_ra) < size:
        if int(len(LowCad_ra) / (182.5 * entries_per_day)) % 2 == 0:
            LowCad_ra.append(ra_values_first[i_first])
            LowCad_dec.append(dec_values_first[i_first])
            i_first = (i_first + 1) % len(ra_values_first)
        else:
            LowCad_ra.append(ra_values_second[i_second])
            LowCad_dec.append(dec_values_second[i_second])
            i_second = (i_second + 1) % len(ra_values_second)

    return np.array(LowCad_ra), np.array(LowCad_dec)


def generate_field_coordinates_option2(start_day, end_day, time_step, observation_hours, pause_start_hour):
    """
    Generates RA and Dec coordinates for survey fields. Use the second option in the currently discussed approach.
    Args:
        start_day:
        end_day:
        timestep:
        observation hours:
        cadence:

    Returns:
        tuple: Arrays of Ra and Dec coordinates
    """

    # Pfad anpassen, falls sich die Datei in einem anderen Verzeichnis befindet
    df = pd.read_csv("Data/LCS_nonoverlapping_grid.csv", sep=",") 

    df = df[(df["V45"] == 1) & (df["A_U<1"] == 1)]
    
    days = np.arange(start_day, end_day, 1)
    daily_times = []

    for day in days:
        for quarter_hour in range(int(1/time_step)):  # quarter-hours in a day
            hour_of_day = quarter_hour *(time_step*24)
            if not (pause_start_hour <= hour_of_day < pause_start_hour + (24 - observation_hours)):
                daily_times.append(day + quarter_hour * time_step)
    
    size = len(daily_times)

    df["Dec_rounded"]=df["Dec"].round(-1)
    df_sorted = df.sort_values(by=['Dec_rounded',"RA"])
    n = 10 
    num_clusters = 8

    sub_dfs = [df_sorted.iloc[i*n:(i+1)*n] for i in range(num_clusters)]
    LowCad_ra=[]
    LowCad_dec=[]

    while len(LowCad_ra) < size:
        for i in range(len(sub_dfs)):
            for k in range(45): #45 days cycle
                for l in sub_dfs[i].index:
                    LowCad_ra.append(sub_dfs[i]["RA"][l])
                    LowCad_dec.append(sub_dfs[i]["Dec"][l])
                
    return np.array(LowCad_ra[:size]), np.array(LowCad_dec[:size])










def generate_field_coordinates_symmetric(start_day, end_day, time_step, observation_hours, pause_start_hour,cadence):
    """
    Generates RA and Dec coordinates for survey fields.

    Args:
        num_fields (int): Total number of fields.
        footprint_diameter (float): Diameter of each field in degrees.

    Returns:
        tuple: Arrays of RA and Dec coordinates.
    """

    days = np.arange(start_day, end_day, 1)
    daily_times = []

    for day in days:
        for quarter_hour in range(96):  # 96 quarter-hours in a day
            hour_of_day = quarter_hour / 4
            if not (pause_start_hour <= hour_of_day < pause_start_hour + (24 - observation_hours)):
                daily_times.append(day + quarter_hour * time_step)
                

    
    size = len(daily_times)
    num_fields=round(cadence*observation_hours/(time_step*24)) #4 -> quarter hours
    print("num_fields="+str(num_fields))
    ra_values= np.concatenate([np.linspace(20,340,int(num_fields/2)),np.linspace(30,330,int(num_fields/3)),np.linspace(40,320,int(num_fields/6))])
    dec_values=sum([int(num_fields/2)*[50],int(num_fields/3)*[60],int(num_fields/6)*[70]],[])
    
    #ra_values= np.concatenate([np.linspace(20,340,int(num_fields/4)),np.linspace(30,330,int(num_fields/6)),np.linspace(40,320,int(num_fields/12)),
    #                           np.linspace(20,340,int(num_fields/4)),np.linspace(30,330,int(num_fields/6)),np.linspace(40,320,int(num_fields/12))])
    #dec_values=sum([int(num_fields/4)*[50],int(num_fields/6)*[60],int(num_fields/12)*[70],
    #                int(num_fields/4)*[-50],int(num_fields/6)*[-60],int(num_fields/12)*[-70]],[])


    # cut to the wanted number
    raLowCadence = ra_values[:num_fields]
    decLowCadence = dec_values[:num_fields]
    
    LowCad_ra = []
    LowCad_dec = []

    # Wiederhole die Felder, bis die Zielgröße erreicht ist
    while len(LowCad_ra) < size:
        LowCad_ra.extend(raLowCadence)
        LowCad_dec.extend(decLowCadence)
        
    return np.array(LowCad_ra[:size]), np.array(LowCad_dec[:size])


def prepare_survey(data, footprint):
    """
    Prepares the ULTRASAT survey data and registers the survey.

    Args:
        data (dict): Survey data including MJD, band, RA, and Dec.
        footprint (shapely.geometry.Polygon): Survey footprint.

    Returns:
        skysurvey.Survey: Configured ULTRASAT survey object.
    """

    return Survey.from_pointings(data, footprint=footprint)

def deg_to_rad(deg):
    return deg * np.pi / 180.0

def calculate_radial_offset(ra, dec, ra2, dec2):
    ra1 = deg_to_rad(ra)
    dec1 = deg_to_rad(dec)
    ra2 = deg_to_rad(ra2)
    dec2 = deg_to_rad(dec2)
    
    cos_angle = (np.sin(dec1) * np.sin(dec2) +
                 np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    
    # Clip it to the area [-1, 1] 
    cos_angle = np.clip(cos_angle,-1,1)
    angle_rad = np.arccos(cos_angle)
    angle_deg = angle_rad * 180.0 / np.pi
    
    return angle_deg

def fieldid_to_ra(fieldid):
    nside=200
    theta,phi=hp.pix2ang(nside,fieldid)
    return round(180-np.degrees(phi),4)

def fieldid_to_dec(fieldid):
    nside=200
    theta,phi=hp.pix2ang(nside,fieldid)
    return round(90-np.degrees(theta),4)

def fieldid_to_offset(fieldid,ra,dec):
    nside=200
    theta,phi=hp.pix2ang(nside,fieldid)
    return calculate_radial_offset(ra,dec,180-np.degrees(phi),90-np.degrees(theta))

def zp_func(x):
    return -0.0009710343981592274 * (x**2) - 0.01175220195794804 * x + 28.274194689718108 #Fitted from Yossis Data

def zp_func_to_radoffset(zp):
    a = -0.0009710343981592274
    b = -0.01175220195794804
    c = 28.274194689718108
    
    discriminant = b**2 - 4*a*(c - zp)
    radoffset = (-b - np.sqrt(discriminant)) / (2 * a)
    
    return radoffset


def maglim_func_fit(x):
    return 0.0008986513787289712 * (x**4) - 0.026248428814867934 * (x**3) + \
           0.1757136381075591 * (x**2) - 0.24756958259309444 * x + 22.40293341329044 #Fitted from Yossis Data. For A0 V.

def maglim_func(x, source_number=41):
    with open('Data/Rdeg.dat', 'r') as f:
        rdeg_data = np.array([float(line.strip()) for line in f])
    
    with open('Data/LimMag.dat', 'r') as f:
        for _ in range(source_number-1):  # skip 40 lines, 41 is Blackbody 20.000K
            f.readline()
        called_line = f.readline().strip()
        LimMag_data = np.array([float(value) for value in called_line.split(',')])
    import matplotlib.pyplot as plt

    #print(rdeg_data)
    #print(LimMag_data)
    #return interp1d(rdeg_data[:-1], LimMag_data[1:], kind="linear", bounds_error=False, fill_value="extrapolate")(x) #Needed to make the wrong limiting magnitude on SNR Calc work!
    return interp1d(rdeg_data, LimMag_data, kind="linear", bounds_error=False, fill_value="extrapolate")(x)
    

rdeg_values = np.array([
    0.00, 0.42, 0.84, 1.27, 1.69, 2.11, 2.53, 2.95, 3.38, 3.80, 
    4.22, 4.64, 5.06, 5.49, 5.91, 6.33, 6.75, 7.17, 7.60, 8.02, 
    8.44, 8.86, 9.28, 9.71, 10.13
])

def find_closest_band_fast(radial_offsets):
    """Findet das nächstgelegene Band für ein Array von radialen Offsets."""
    differences = np.abs(radial_offsets[:, np.newaxis] - rdeg_values)
    closest_indices = np.argmin(differences, axis=1)
    closest_rdeg = rdeg_values[closest_indices]
    band_names = [f"ultrasat_band_{r:.2f}" for r in closest_rdeg]
    return band_names

def add_survey_properties(df, source_number=41):
    """
    Adds calculated properties (RadOffset, zp, skynoise, and band) to the survey data.

    Args:
        df (pd.DataFrame): Survey data.
        source (int): Source index for magnitude limit calculation. Default is 41.

    Returns:
        pd.DataFrame: Updated survey data with additional columns.
    """
    results = []
    chunk_size=50000
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size].copy()
        chunk["RadOffset"] = chunk.apply(lambda row: fieldid_to_offset(row["fieldid"], row["ra"], row["dec"]), axis=1)
        chunk["zp"] = zp_func(chunk["RadOffset"].values)
        chunk["skynoise"] = 10**(0.4 * (chunk["zp"] - maglim_func(chunk["RadOffset"].values,source_number))) / 5
        closest_bands = find_closest_band_fast(chunk["RadOffset"].values)
        chunk["band"] = closest_bands
        results.append(chunk)
    return pd.concat(results)


def add_survey_properties2(df):
    """
    Adds calculated properties (zero-point, skynoise, and band) to the survey data.

    Args:
        df (pd.DataFrame): Survey data.
        zp_func (callable): Function to calculate zero-point from radial offset.
        maglim_func (callable): Function to calculate magnitude limit.
        find_closest_band_func (callable): Function to assign the closest band based on radial offset.

    Returns:
        pd.DataFrame: Updated survey data with additional columns.
    """
    # Calculate Radial Offset
    print("Radoffset")
    df["RadOffset"] = fieldid_to_offset(df["fieldid"].values, df["ra"].values, df["dec"].values)

    # Calculate Zero-Point
    print("point")
    df["zp"] = zp_func(df["RadOffset"].values)

    # Calculate Sky Noise
    print("noise")
    df["skynoise"] = 10**(0.4 * (df["zp"] - maglim_func(df["RadOffset"].values))) / 5

    # Assign Closest Band
    print("band")
    df["band"] = find_closest_band_fast(df["RadOffset"].values)

    return df



def render_healpy_to_matplotlib(data, title=None, vmin=None, vmax=None, **kwargs):
    """
    Render a healpy.mollview map into a matplotlib figure.

    Parameters
    ----------
    data : array-like
        The data to plot using healpy.mollview.
    title : str, optional
        Title of the plot.
    vmin : float, optional
        Minimum value for the color scale.
    vmax : float, optional
        Maximum value for the color scale.
    **kwargs : dict
        Additional arguments passed to healpy.mollview.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure containing the healpy map.
    """
    fig = plt.figure(figsize=(10, 6))
    hp.mollview(data, title=title, fig=fig.number, **kwargs)
    hp.graticule()  # Add gridlines
    return fig

