import yaml
import os
import hashlib
import pickle
import warnings
import skysurvey
import skysurvey.survey.polygon as surveypolygon
import pandas as pd
import matplotlib.pyplot as plt
import sncosmo
import numpy as np
from datetime import datetime
from multiprocessing import Pool
warnings.filterwarnings("ignore", category=FutureWarning, module="skysurvey")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "Modules"))


from models import create_salt3_model
from simulation import simulate_blackbody_events, plot_snia_data, convert_drawn_data_to_instance
from filters import register_ultrasat_bands
from survey_plan import (
    create_footprint,
    generate_time_array,
    generate_field_coordinates_option1,
    generate_field_coordinates_option2,
    prepare_survey,
    add_survey_properties,
    maglim_func
)
from lightcurves import initialize_dataset,stack_high_cadence_data, process_lightcurve_data
from plotting import extract_data_for_plotting, plot_survey_overview, generate_unique_filename



def main():
    print("main() function is starting...")
    # Load the configuration file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)


    # Create the SALT3 model
    try:
        model = create_salt3_model(config["model_directory"])
        print("SALT3 model successfully created and registered!")
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
    temperatures=[2000,4000,6000,8000,10000,20000,30000,40000]
    sniadata,sniamodel = simulate_blackbody_events(start_time=start_time,duration=duration,temperature=temperatures[source_number-36])
    print(f"Simulated {len(sniadata)} events passing the magnitude filter.")

    # Convert the drawn data to model instance
    print("Convert drawn data to model instance..")
    #sniainstance = convert_drawn_data_to_instance(sniamodel, sniadata,sniamodel)
    sniainstance = convert_drawn_data_to_instance(template_name, sniadata,sniamodel)
    print(sniainstance.data.head(10))
    lbdas=np.array(np.arange(1000.,10000.,1))
    plt.plot(lbdas,sniamodel.get_spectrum(0,lbdas=lbdas))
    plt.show()
    
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
        'LC_cadence': LC_cadence,
        "Alternative_Survey": Alternative_Survey,
        "source_number": source_number
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
                    pause_start_hour=pause_start_hour,
                    cadence=LC_cadence
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
            "gain": 2, #2 is given value, might be specified soon
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
    survey.data["gain"]=1

    print("Survey preparation complete. Displaying sample data:")
    print(survey.data.head(10))

    #_____________________________________________________
    #LIGHTCURVES

    lightcurve_config=config["lightcurves"]
    HC_stacking=lightcurve_config["HC_stacking"]
    HC_stack_number=lightcurve_config["HC_stack_number"]
    min_sn_ratio=lightcurve_config["min_sn_ratio"]
    min_sn_points=1
    min_detections=1
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




    df_ultrasat_lightcurves=ultrasat_lightcurves.data.copy()
    multiindex_values = df_ultrasat_lightcurves.index.get_level_values(0)
    df_ultrasat_lightcurves["z"] = multiindex_values.map(
        lambda x: sniainstance.data.iloc[x]["z"] if x < len(sniainstance.data) else None
    )


    ultrasat_lightcurves.set_data(df_ultrasat_lightcurves)
    
    # Display a preview of the processed data
    print("Processed dataset preview:")
    print(ultrasat_lightcurves.data.head(10))
    
    # Saving the lightcurves to the created folder
    print("Saving the lightcurve")
    ultrasat_lightcurves.data.to_parquet((f"Lightcurves/lightcurves/file_lightcurves_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.parquet"))

    #Filter the data, so we only have values close to the peak
    filtered_lightcurves = df_ultrasat_lightcurves[(df_ultrasat_lightcurves['phase'] >= -0.2) & (df_ultrasat_lightcurves['phase'] <= 0.2)]

    filtered_indices = filtered_lightcurves.index.get_level_values(0)
    sniadata = sniadata.reset_index()  # Zeilennummer als Spalte hinzufügen
    sniadata.rename(columns={'index': 'row_number'}, inplace=True) 
    print(sniadata.head())
    filtered_BB = sniadata[sniadata['row_number'].isin(filtered_indices)]

    #merge the columns 
    combined_df = pd.merge(
        filtered_lightcurves.reset_index()[['index',"mag","magerr", "flux", "fluxerr", "band"]],
        filtered_BB.rename(columns={'row_number': 'index'})[["index", "US-Band", "magobs"]],
        on='index'
    )
    combined_df["SNR"]=combined_df["flux"]/combined_df["fluxerr"]
    combined_df['RadOff'] = combined_df['band'].str.extract(r'(\d+\.\d+)$').astype(float)
    combined_df["LimMag"] = combined_df["RadOff"].apply(lambda rad_off: maglim_func(rad_off, source_number=source_number))

    # show results
    print(combined_df)
    combined_df.to_parquet(f"Sources/"+str(source_number)+"_file.parquet")



    


   




if __name__ == "__main__":
    main()

#if __name__ == "__main__":
#    for i in range(20):
#        print(f"run {i + 1}:")
#        main()