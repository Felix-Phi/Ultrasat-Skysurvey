import sncosmo
import numpy as np
import pandas as pd
from skysurvey import DataSet
import skysurvey
from tqdm import tqdm

def initialize_dataset(snia_data, survey):
    """
    Initializes the light curve dataset using supernova and survey data.

    Args:
        snia_data (pandas.DataFrame): Data of simulated supernovae.
        survey (skysurvey.Survey): Survey object.

    Returns:
        skysurvey.DataSet: Initialized dataset with light curves.
    """
    dataset = DataSet.from_targets_and_survey(snia_data, survey)
    return dataset



def stack_high_cadence_data(dataset, n=12):
    """
    Stacks high-cadence survey data over specified intervals (e.g., 3 hours).

    Args:
        dataset (skysurvey.DataSet): Original high-cadence dataset.
        n (int): Number of rows to group together for stacking.

    Returns:
        skysurvey.DataSet: Dataset with stacked data.
    """
    # Group the data into intervals of size `n`
    grouped = dataset.data.groupby(np.arange(len(dataset.data)) // n)

    # Aggregate values for the stacked dataset
    result = grouped.agg({
        'flux': 'mean',             # Mean of flux
        'time': 'mean',             # Mean of time
        'fluxerr': lambda x: np.sqrt(np.sum(x**2)) / len(x) if len(x) > 0 else np.nan  # Correct error propagation
    }).reset_index(drop=True)

    # Add additional columns with consistent values
    result['fieldid'] = grouped['fieldid'].agg(lambda x: x.mode()[0]).reset_index(drop=True)
    result['band'] = grouped['band'].agg(lambda x: x.mode()[0]).reset_index(drop=True)
    result['zp'] = grouped['zp'].agg(lambda x: x.mode()[0]).reset_index(drop=True)
    result['zpsys'] = grouped['zpsys'].agg(lambda x: x.mode()[0]).reset_index(drop=True)

    # Retain original indices for reference
    result['original_index'] = [group.index[0] for _, group in grouped]

    # Extract and set new indices
    result[['index', '']] = pd.DataFrame(result['original_index'].tolist(), index=result.index)
    result.set_index(['index', ''], inplace=True)

    # Drop the redundant column and reorder columns
    result.drop(columns=['original_index'], inplace=True)
    result = result[['fieldid', 'time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys']]

    # Update the dataset with the stacked data
    dataset.set_data(result)

    return dataset

def filter_data(dataset,simulation, min_sn_ratio=1, min_sn_points=3, min_detections=5, max_phase=20):
    """
    Filters the dataset based on multiple criteria:
    1. Remove all data points with S/N < min_sn_ratio.
    2. Keep only supernovae with at least min_detections points where S/N ≥ min_sn_points.
    3. Remove data points with phase values outside [-max_phase, max_phase].

    Args:
        dataset (skysurvey.DataSet): Original dataset with light curve data.
        min_sn_ratio (float): Minimum S/N ratio for data point filtering.
        min_sn_points (float): Minimum S/N for counting valid points in supernovae.
        min_detections (int): Minimum number of valid detections per supernova.
        max_phase (float): Maximum absolute phase value (days) for filtering.

    Returns:
        pandas.DataFrame: Filtered dataset.
    """
    # Start with the dataset's raw data
    data = dataset.data

    # Step 1: Remove data points with S/N < min_sn_ratio
    data = data[data["flux"] / data["fluxerr"] >= min_sn_ratio]

    # Step 2: Keep only supernovae with at least `min_detections` points where S/N >= `min_sn_points`
    valid_sn_points = (
        data[data["flux"] / data["fluxerr"] >= min_sn_points]
        .groupby(dataset._data_index)
        .size()
    )
    valid_indices = valid_sn_points[valid_sn_points >= min_detections].index
    data = data[data.index.get_level_values("index").isin(valid_indices)]

    # Step 3: Add peak times and calculate phases
    peak_times = data.index.map(lambda x: simulation.data["t0"][x[0]])
    data["peak_time"] = peak_times
    data["phase"] = data["time"] - data["peak_time"]

    # Step 4: Remove data points with phase outside [-max_phase, max_phase]
    data = data[abs(data["phase"]) < max_phase]

    return data


def calculate_magnitude(flux, zp):
    """
    Calculates the magnitude from flux and zero-point.

    Args:
        flux (float): Observed flux.
        zp (float): Zero-point for magnitude calculation.

    Returns:
        float: Calculated magnitude.
    """
    return -2.5 * np.log10(flux) + zp


def calculate_magnitude_error(row):
    """ 
    Calculates the magnitude error from flux and flux error.

    Args:
        row (pandas.Series): Row of the DataFrame containing flux and fluxerr.

    Returns:
        float: Magnitude error.
    """
    flux = row["flux"]
    fluxerr = row["fluxerr"]
    return ((-2.5*fluxerr/flux/np.log(10))**2)**0.5


def process_lightcurve_data(dataset,simulation, min_sn_ratio=1, min_sn_points=3, min_detections=5, max_phase=20):
    """
    Processes light curve data by filtering and adding calculated columns.

    Args:
        dataset (skysurvey.DataSet): Light curve dataset.
        min_sn_ratio (float): Minimum S/N ratio for filtering data points.
        min_sn_points (float): Minimum S/N for counting valid detections per supernova.
        min_detections (int): Minimum number of valid detections per supernova.
        max_phase (float): Maximum absolute phase value for filtering.

    Returns:
        skysurvey.DataSet: Updated dataset with filtered data.
    """
    # Apply filters to the data
    filtered_data = filter_data(dataset,simulation, min_sn_ratio, min_sn_points, min_detections, max_phase)

    # Add calculated magnitudes and errors
    filtered_data["mag"] = filtered_data.apply(
        lambda row: calculate_magnitude(row["flux"], row["zp"]), axis=1
    )
    filtered_data["magerr"] = filtered_data.apply(calculate_magnitude_error, axis=1)

    # Update the dataset with the filtered data
    dataset.set_data(filtered_data)

    return dataset

#__________________________________________________
#RATES

def filter_data_rates(dataset,min_sn_ratio=1, min_sn_points=3, min_detections=5, max_phase=20):
    """
    Filters the dataset based on multiple criteria:
    1. Remove all data points with S/N < min_sn_ratio.
    2. Keep only supernovae with at least min_detections points where S/N ≥ min_sn_points.
    3. Remove data points with phase values outside [-max_phase, max_phase].

    Args:
        dataset (skysurvey.DataSet): Original dataset with light curve data.
        no simulation, because the dataset include the peak times already.
        min_sn_ratio (float): Minimum S/N ratio for data point filtering.
        min_sn_points (float): Minimum S/N for counting valid points in supernovae.
        min_detections (int): Minimum number of valid detections per supernova.
        max_phase (float): Maximum absolute phase value (days) for filtering.

    Returns:
        pandas.DataFrame: Filtered dataset.
    """
    # Start with the dataset's raw data
    data = dataset.data

    # Step 1: Remove data points with S/N < min_sn_ratio
    data = data[data["flux"] / data["fluxerr"] >= min_sn_ratio]

    # Step 2: Keep only supernovae with at least `min_detections` points where S/N >= `min_sn_points`
    valid_sn_points = (
        data[data["flux"] / data["fluxerr"] >= min_sn_points]
        .groupby(dataset._data_index)
        .size()
    )
    valid_indices = valid_sn_points[valid_sn_points >= min_detections].index
    data = data[data.index.get_level_values("index").isin(valid_indices)]

    # Step 3: calculate phases
    data["phase"] = data["time"] - data["peak_time"]

    # Step 4: Remove data points with phase outside [-max_phase, max_phase]
    data = data[abs(data["phase"]) < max_phase]

    #Step 5: add explosion time
    data["explosion_phase"] = np.nan
    data["time_after_explosion"] = np.nan

    # Group the DataFrame by the first index level
    cache = {}
    precision = 3  # Rundung auf 3 Dezimalstellen

    # Gruppiere nach dem ersten Level des Multiindex
    groups = list(data.groupby(level=0))
    for first_idx, group in tqdm(groups, desc="Verarbeite Gruppen"):
        rep = group.iloc[0]
        # Erstelle einen Schlüssel, indem du die Parameter rundest
        key = (round(rep["z"], precision),
            round(rep["x0"], precision),
            round(rep["x1"], precision),
            round(rep["c"], precision),
            round(rep["zp"], precision))
        
        if key not in cache:
            times = np.linspace(-20, -10, 1000)
            model = sncosmo.Model(source='QinanSalt3')
            model.set(z=rep["z"], x0=rep["x0"], x1=rep["x1"], c=rep["c"])
            flux = model.bandflux('ztfg', times, zp=rep["zp"], zpsys='ab')
            dflux_dt = np.gradient(flux, times)
            threshold = 1e-1
            positive_deriv_indices = np.where(dflux_dt > threshold)[0]
            if len(positive_deriv_indices) == 0:
                explosion_phase = np.nan
            else:
                explosion_phase = times[positive_deriv_indices[0]]
            cache[key] = explosion_phase
        else:
            explosion_phase = cache[key]
        
        # Weise den berechneten explosion_phase allen Zeilen in der Gruppe zu
        data.loc[first_idx, "explosion_phase"] = explosion_phase
        # Berechne time_after_explosion individuell für jede Zeile
        data.loc[first_idx, "time_after_explosion"] = group["phase"] - explosion_phase





    return data


def process_lightcurve_data_rates(dataset, min_sn_ratio=1, min_sn_points=3, min_detections=5, max_phase=20):
    """
    Processes light curve data by filtering and adding calculated columns.

    Args:
        dataset (skysurvey.DataSet): Light curve dataset.
        no simulation, because the dataset include the peak times already.
        min_sn_ratio (float): Minimum S/N ratio for filtering data points.
        min_sn_points (float): Minimum S/N for counting valid detections per supernova.
        min_detections (int): Minimum number of valid detections per supernova.
        max_phase (float): Maximum absolute phase value for filtering.

    Returns:
        skysurvey.DataSet: Updated dataset with filtered data.
    """
    # Apply filters to the data
    filtered_data = filter_data_rates(dataset, min_sn_ratio, min_sn_points, min_detections, max_phase)

    # Add calculated magnitudes and errors
    filtered_data["mag"] = filtered_data.apply(
        lambda row: calculate_magnitude(row["flux"], row["zp"]), axis=1
    )
    filtered_data["magerr"] = filtered_data.apply(calculate_magnitude_error, axis=1)

    # Update the dataset with the filtered data
    dataset.set_data(filtered_data)

    return dataset


