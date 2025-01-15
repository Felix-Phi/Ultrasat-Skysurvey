import pandas as pd
import copy
from skysurvey import ZTF


def create_ztf_survey(start_time, duration):
    """
    Create and filter the ZTF survey based on the given time range.
    The logs have to be in the folder "Data".

    Args:
        start_time (float): Start time of the survey (MJD).
        duration (float): Duration of the survey in days.

    Returns:
        skysurvey.ZTF: Filtered ZTF survey object.
    """
    # Load ZTF survey from logs
    ztfsurvey = ZTF.from_logs()

    # Filter ZTF survey data by time range
    dfztfsurvey = ztfsurvey.data
    dfztfsurvey = dfztfsurvey[
        (dfztfsurvey["mjd"] >= start_time) & (dfztfsurvey["mjd"] <= start_time + duration)
    ]
    ztfsurvey.set_data(dfztfsurvey)

    return ztfsurvey


def filter_ztf_data(dataset, min_sn_points=1, min_detections=5):
    """
    Filter ZTF dataset based on signal-to-noise ratio and number of detections.

    Args:
        dataset (skysurvey.DataSet): ZTF dataset.
        min_sn_ratio (float): Minimum S/N ratio for data points.
        min_detections (int): Minimum number of valid detections per supernova.

    Returns:
        skysurvey.DataSet: Filtered ZTF dataset.
    """
    data = dataset.data

    # Filter by signal-to-noise ratio
    data = data[data["flux"] / data["fluxerr"] >= min_sn_points]

    # Filter by minimum number of detections
    valid_counts = data.groupby(dataset._data_index).size()
    valid_indices = valid_counts[valid_counts >= min_detections].index
    data = data[data.index.get_level_values("index").isin(valid_indices)]

    dataset.set_data(data)
    return dataset


def combine_surveys(ztf_dataset, ul_dataset):
    """
    Combine ZTF and ULTRASAT datasets.

    Args:
        ztf_dataset (skysurvey.DataSet): ZTF dataset.
        ul_dataset (skysurvey.DataSet): ULTRASAT dataset.

    Returns:
        skysurvey.DataSet: Combined dataset.
    """
    combined_dataset = copy.deepcopy(ztf_dataset)
    combined_data = pd.concat([ztf_dataset.data, ul_dataset.data])
    combined_dataset.set_data(combined_data)

    return combined_dataset


def extract_observation_indices(dataset, bands):
    """
    Extract unique indices of observations for specific bands.

    Args:
        dataset (pandas.DataFrame): Combined survey dataset.
        bands (list): List of band names to filter.

    Returns:
        list: Unique indices of observations in the specified bands.
    """
    filtered_indexes = dataset[dataset["band"].isin(bands)].index
    unique_indices = []
    for i in filtered_indexes:
        if i[0] not in unique_indices:
            unique_indices.append(i[0])

    return unique_indices


def find_combined_indices(us_indices, ztf_indices):
    """
    Find combined indices that are observed in both ULTRASAT and ZTF.

    Args:
        us_indices (list): ULTRASAT observation indices.
        ztf_indices (list): ZTF observation indices.

    Returns:
        list: Indices that are observed in both surveys.
    """
    return list(set(us_indices) & set(ztf_indices))


def sort_combined_indices_by_mag(combined_indices, mag_data):
    """
    Sort combined indices by magnitude.

    Args:
        combined_indices (list): Combined indices from both surveys.
        mag_data (pandas.DataFrame): DataFrame containing indices and magnitudes.

    Returns:
        list: Sorted combined indices based on magnitude.
    """
    filtered_mag = mag_data[mag_data["index"].isin(combined_indices)]
    sorted_indices = filtered_mag.sort_values(by="mag", ascending=True)
    return sorted_indices["index"].tolist()
