import yaml
import os
import hashlib
import pickle
import warnings
import matplotlib.pyplot as plt
import skysurvey
import pandas as pd
from datetime import datetime
warnings.filterwarnings("ignore", category=FutureWarning, module="skysurvey")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "Modules"))


from templates import create_salt3_template_dust
from simulation import simulate_snia_events_dust, plot_snia_data_dust, convert_drawn_data_to_instance
from filters import register_ultrasat_bands
from survey_plan import (
    create_footprint,
    generate_time_array,
    generate_field_coordinates_option1,
    generate_field_coordinates_option2,
    prepare_survey,
    add_survey_properties,
    zp_func_to_radoffset
)
from lightcurves import initialize_dataset,stack_high_cadence_data, process_lightcurve_data
from plotting import extract_data_for_plotting, plot_survey_overview
from ZTF_survey import (
    create_ztf_survey,
    filter_ztf_data,
    combine_surveys,
    extract_observation_indices,
    find_combined_indices,
    sort_combined_indices_by_mag,
)
from curve_fit import build_results_table_dust, create_comparison_table_dust


def main():
    print("main() function is starting...")
    # Load the configuration file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)


    # Create the SALT3 model
    try:
        modeldust = create_salt3_template_dust(config["template_directory"])
        print("SALT3 model with dust effect successfully created and registered!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        # Register ULTRASAT bands
    filter_config = config["filters"]
    register_ultrasat_bands(
        transmission_file=filter_config["transmission_file"],
        wavelength_file=filter_config["wavelength_file"],
        rdeg_file=filter_config["rdeg_file"]
    )


    
    #_____________________________________________________________________
    #SIMULATION

    # Extract simulation parameters
    sim_config = config["simulation"]
    template_name = sim_config["template_name"]
    redshift_max = sim_config["redshift_max"]
    start_time = sim_config["start_time"]
    duration = sim_config["duration"]
    dust_extinction=sim_config["dust_extinction"]
    magnitude_limit = sim_config["magnitude_limit"]
    plot_results = sim_config["plot_results"]
    source_number=sim_config["source_number"]

    # Simulate SNIa events
    print("Simulating SNIa events...")
    sniadata,sniamodel = simulate_snia_events_dust(
        template_name=template_name,
        redshift_max=redshift_max,
        start_time=start_time,
        duration=duration,
        dust_extinction=dust_extinction,
        magnitude_limit=magnitude_limit,
    )
    print(f"Simulated {len(sniadata)} events passing the magnitude filter.")

    # Optionally plot simulated data
    if plot_results:
        print("Plotting SNIa data...")
        plot_snia_data_dust(sniamodel, redshift_max, duration,plot_show=False)

    # Convert the drawn data to model instance
    print("Convert drawn data to model instance..")
    sniainstance = convert_drawn_data_to_instance(modeldust, sniadata,sniamodel)
    print("SNIa instance parameters:")
    print(sniainstance.template_parameters)
    print(sniainstance.data.head(10))
    
    #_________________________________________________________-
    #SURVEY PLAN

    survey_config = config["survey"]
    HighCadence=survey_config["HighCadence"]
    time_step=survey_config["time_step"]
    slew_time=survey_config["slew_time"]
    pause_start_hour=survey_config["pause_start_hour"]
    observation_hours=survey_config["observation_hours"]
    LC_cadence=survey_config["LC_cadence"]
    Alternative_Survey=survey_config["Alternative_Survey"]

    #Check if we already have the survey in the cache.
    #It is repetitive and takes a lot of time to compute.
    #For that we create a hash for the parameters.

    if HighCadence:
        parameters = {
        'start_time': start_time,
        'duration': duration,
        'HighCadence': HighCadence,
        'time_step': time_step,
        'pause_start_hour': pause_start_hour,
        'observation_hours': observation_hours,
        "source_number": source_number
        }
        hilocadence="High Cadence"
    else:
        parameters = {
        'start_time': start_time,
        'duration': duration,
        'HighCadence': HighCadence,
        'time_step': time_step+slew_time,
        'pause_start_hour': pause_start_hour,
        'observation_hours': observation_hours,
        "source_number": source_number,
        'LC_cadence': LC_cadence,
        "Alternative_Survey": Alternative_Survey
        }
        hilocadence="Low Cadence"

    # Serialize the parameters to create a unique identifier
    param_string = str(sorted(parameters.items()))
    param_hash = hashlib.md5(param_string.encode('utf-8')).hexdigest()

    cache_dir="Cache"
    cache_filename = os.path.join(cache_dir, f"survey_{param_hash}.parquet")

    if os.path.exists(cache_filename):
        print("Cached survey data found. Loading survey from file...")
        survey_df=pd.read_parquet(cache_filename)
        survey=skysurvey.Survey()
        survey.set_data(survey_df)

    else:
        print("No cached survey data found. Generating survey...")
        #Create to footprint
        print("Creating ULTRASAT footprint...")
        footprint = create_footprint()


        # Generate observation time array
        print("Generating observation time array...")


        if HighCadence:
            mjd_times = generate_time_array(
                    start_day=start_time,
                    end_day=start_time+duration,
                    time_step=time_step,
                    observation_hours=observation_hours,
                    pause_start_hour=pause_start_hour,
                )
        
            #HC observes only one spot.
            ra=[57]*len(mjd_times)
            dec=[-47]*len(mjd_times)


        else:
            mjd_times = generate_time_array(
                    start_day=start_time,
                    end_day=start_time+duration,
                    time_step=time_step+slew_time,
                    observation_hours=observation_hours,
                    pause_start_hour=pause_start_hour,
                )
            if Alternative_Survey:
                # Generate RA and Dec field coordinates for the LC in second option survey plan.
                print("Generating field coordinates...")
                ra, dec = generate_field_coordinates_option2(
                    start_day=start_time,
                    end_day=start_time+duration,
                    time_step=time_step+slew_time,
                    observation_hours=observation_hours,
                    pause_start_hour=pause_start_hour
                )
            else:
                # Generate RA and Dec field coordinates for the LC in first option survey.
                print("Generating field coordinates...")
                ra, dec = generate_field_coordinates_option1(
                    start_day=start_time,
                    end_day=start_time+duration,
                    time_step=time_step+slew_time,
                    observation_hours=observation_hours,
                    pause_start_hour=pause_start_hour,
                    cadence=LC_cadence
                )

        # Create survey data structure
        print("Create survey data structure...")
        data = {
            "mjd": mjd_times,
            "band": "ultrasat",  #just some band, change that in the next step
            "ra": ra,
            "dec": dec,
            "gain": 2, #given value, might be specified soon
            "zp": 0, #changed in the next step
            "skynoise": 0, #changed in the next step
        }
        survey = prepare_survey(data, footprint=footprint)
        

        # Add calculated properties to the survey data
        print("Adding calculated properties to the survey data...")
        survey_df = survey.data.copy()
        survey_df = add_survey_properties(survey_df,source_number=source_number)

        survey.set_data(survey_df)

        survey.show()
        fig = plt.gcf()  # Get current figure
        fig.savefig(cache_filename+"_sky_coverage.png", dpi=300)

        # Save survey to cache file

        print("Saving survey to cache file...")
        survey_df.to_parquet(cache_filename)

    # Display results
    #print("Survey preparation complete. Displaying sample data:")
    #print(survey.data.head(10))

    #_____________________________________________________
    #LIGHTCURVES

    lightcurve_config=config["lightcurves"]
    HC_stacking=lightcurve_config["HC_stacking"]
    HC_stack_number=lightcurve_config["HC_stack_number"]
    min_sn_ratio=lightcurve_config["min_sn_ratio"]
    min_sn_points=3
    min_detections=3
    max_phase=lightcurve_config["max_phase"]
    plot_overview=lightcurve_config["plot_overview"]
    plot_show=lightcurve_config["plot_show"]


    # Initialize the lightcurve dataset
    print("Initializing the lightcurve dataset...")
    dataset = initialize_dataset(sniainstance, survey)

    # Apply high-cadence stacking
    if HighCadence and HC_stacking:
        print("Stacking data over "+str(round(time_step*24*HC_stack_number,1))+"-hour intervals...")
        dataset = stack_high_cadence_data(dataset, n=HC_stack_number)

    # Process the dataset (filter, add calculated columns)
    print("Processing light curve data...")
    ultrasat_lightcurves = process_lightcurve_data(
        dataset=dataset,simulation=sniainstance, min_sn_ratio=min_sn_ratio, min_sn_points=min_sn_points, min_detections=min_detections, max_phase=max_phase
    )

    # Display a preview of the processed data
    print("Processed dataset preview:")
    print(ultrasat_lightcurves.data.head(10))


    # _____________________________________________________
    # PLOTTING
    if plot_overview:
        print("Extracting data for plotting...")
        filtered_data, highest_mag_list = extract_data_for_plotting(ultrasat_lightcurves, sniainstance)
        #print(highest_mag_list)
        print("Generating and saving survey overview plot...")
        plot_survey_overview(
            data=filtered_data,
            min_sn_points=min_sn_points,
            min_detections=min_detections,
            hilocadence=hilocadence,
            duration=duration,
            observation_hours=observation_hours,
            cadence=LC_cadence, 
            Alternative_Survey=Alternative_Survey,
            output_file=f"survey_Overview_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png",
            plot_show=plot_show,
            folder="Results"
        )
    
    #_______________________________________________________________
    # ZTF_SURVEY

    parameters = {
        'start_time': start_time,
        'duration': duration
        }
    # Serialize the parameters to create a unique identifier
    param_string = str(sorted(parameters.items()))
    param_hash = hashlib.md5(param_string.encode('utf-8')).hexdigest()

    cache_dir="Cache"
    cache_filename = os.path.join(cache_dir, f"ZTF_survey_{param_hash}.pkl")

    if os.path.exists(cache_filename):
        print("Cached ZTF-survey data found. Loading ztf.survey from file...")
        with open(cache_filename, 'rb') as f:
            ztf_survey = pickle.load(f)

    else:
        print("No cached ZTF survey data found. Generating ztf_survey...")
        ztf_survey = create_ztf_survey(start_time=start_time, duration=duration)

        # Save survey_df and parameters to cache file
        print("Saving ztf_survey to cache file...")
        with open(cache_filename, 'wb') as f:
            pickle.dump(ztf_survey, f)

    print(ztf_survey.data.head(10))

    print("Initialize the lightcurve dataset for ZTF...")
    ztf_dataset = initialize_dataset(sniainstance, ztf_survey)
    
    print("Processing the lightcurve data for ZTF...")
    ztf_lightcurves = process_lightcurve_data(
                                           dataset=ztf_dataset,simulation=sniainstance, min_sn_ratio=min_sn_ratio,
                                           min_sn_points=10, min_detections=min_detections, max_phase=100
                                            )



    # COMBINE ZTF AND ULTRASAT DATASETS
    print("Combining ZTF and ULTRASAT surveys...")
    combined_lightcurves = combine_surveys(ztf_lightcurves, ultrasat_lightcurves)

    # Extract observation indices for each survey
    print("Extracting observation indices...")
    ztf_bands = ["ztfr", "ztfg", "ztfi"]
    ultrasat_bands = [
        "ultrasat_band_0.00", "ultrasat_band_0.42", "ultrasat_band_0.84",
        "ultrasat_band_1.27", "ultrasat_band_1.69", "ultrasat_band_2.11",
        "ultrasat_band_2.53", "ultrasat_band_2.95", "ultrasat_band_3.38",
        "ultrasat_band_3.80", "ultrasat_band_4.22", "ultrasat_band_4.64",
        "ultrasat_band_5.06", "ultrasat_band_5.49", "ultrasat_band_5.91",
        "ultrasat_band_6.33", "ultrasat_band_6.75", "ultrasat_band_7.60",
        "ultrasat_band_8.02", "ultrasat_band_7.17"
    ]

    ztf_indices = extract_observation_indices(combined_lightcurves.data, ztf_bands)
    ultrasat_indices = extract_observation_indices(combined_lightcurves.data, ultrasat_bands)

    # Find combined indices and sort by magnitude
    print("Finding combined observation indices...")
    combined_indices = find_combined_indices(ultrasat_indices, ztf_indices)
    combined_indices_sorted_by_mag= sort_combined_indices_by_mag(combined_indices, highest_mag_list)

    print(ztf_lightcurves.data.head(30))

    #_________________________________________________________________
    #FITTING AND COMPARING THE FITS

    print("Fitting light curves for combined and ZTF datasets...")
    results_table = build_results_table_dust(
        combined_lightcurves=combined_lightcurves,
        ztf_lightcurves=ztf_lightcurves,
        sniadata=sniainstance.data,
        sorted_indices=combined_indices_sorted_by_mag,
        model=modeldust,
        rv=1,
        folder="rv1"
    )
    results_table = build_results_table_dust(
        combined_lightcurves=combined_lightcurves,
        ztf_lightcurves=ztf_lightcurves,
        sniadata=sniainstance.data,
        sorted_indices=combined_indices_sorted_by_mag,
        model=modeldust,
        rv=2,
        folder="rv2"
    )
    results_table = build_results_table_dust(
        combined_lightcurves=combined_lightcurves,
        ztf_lightcurves=ztf_lightcurves,
        sniadata=sniainstance.data,
        sorted_indices=combined_indices_sorted_by_mag,
        model=modeldust,
        rv=3.1,
        folder="rv3.1"
    )

    # Display results
    print("Results of the fits (filtered):")
    print(results_table.head(20))

    print("Creating comparison table...")
    comparison_table = create_comparison_table_dust(fit_results_table=results_table)
    
    # Display comparison table preview
    print("Comparison table preview:")
    print(comparison_table.head(10))




if __name__ == "__main__":
    main()