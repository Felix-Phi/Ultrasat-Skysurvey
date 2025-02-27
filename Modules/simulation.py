import numpy as np
import skysurvey
from skysurvey import SNeIa
from skysurvey.effects import Effect
from skysurvey import effects
from skysurvey.tools import blackbody
import sncosmo
import matplotlib.pyplot as plt
from plotting import generate_unique_filename



def simulate_snia_events(template_name, redshift_max, start_time, duration,dust_extinction, magnitude_limit):
    """
    Simulate SNIa events for a given template and redshift range.

    Args:
        template_name (str): Name of the SNIa template to use (e.g., "QinanSalt3").
        redshift_max (float): Maximum redshift for the simulation.
        start_time (float): Start time for the simulation (e.g., MJD).
        duration (float): Duration of the simulation in days.
        magnitude_limit (float): Magnitude limit for filtering events.

    Returns:
        pandas.DataFrame: Simulated SNIa events with additional magnitude data.
    """
    # Initialize SNIa simulation object
    snia = skysurvey.SNeIa()
    snia.set_template(template_name)
    if dust_extinction:
        snia.add_effect(effect=effects.mw_extinction)
    
    # Draw supernova events
    num_events = int(snia.get_rate(redshift_max) / (365/duration))

    sniadata = snia.draw(size=num_events, zmax=redshift_max, zmin=0,
                         tstart=start_time, tstop=start_time + duration, inplace=True)

    # Calculate observed magnitudes in the specified band
    target_magnitudes = []
    for j in range(len(sniadata)):
        mag = snia.get_target_mag(j, band="ultrasat_band_0.00", phase=0, magsys="ab", restframe=True)
        target_magnitudes.append(mag)
    sniadata["US-Band"] = target_magnitudes

    # Apply magnitude filter
    sniadata = sniadata[sniadata["US-Band"] <= magnitude_limit]
    sniadata = sniadata.reset_index(drop=True)
    
    # Take the model for future usage
    sniamodel=snia

    return sniadata,sniamodel

def simulate_blackbody_events(start_time,duration,temperature):
    phase = np.linspace(-100, 100, 50) # phase definition range
    time_scale=5.
    amplitude = 1e-15+phase*0# fast decay
    #print("amplitude ="+str(amplitude))
    temperature = np.linspace(temperature, temperature, len(phase)) # stays at 20.000
    #print("temperature="+str(temperature))
    lbda = np.linspace(800, 10_000, 1000)
    bb_source = blackbody.get_blackbody_transient_source(phase=phase, 
                                                    amplitude=amplitude, 
                                                    temperature=temperature, 
                                                   lbda=lbda) # lbda has a default
    bb_transientmodel = skysurvey.TSTransient(bb_source)
    bb_transientdata=bb_transientmodel.draw(1000, inplace=True, zmax=0.11, tstart=start_time, tstop=start_time+duration)
    target_magnitudes = []
    for j in range(len(bb_transientdata)):
        mag = bb_transientmodel.get_target_mag(j, band="ultrasat_band_0.00", phase=0, magsys="ab", restframe=True)
        target_magnitudes.append(mag)
    bb_transientdata["US-Band"] = target_magnitudes


    return bb_transientdata,bb_transientmodel



def plot_snia_data(sniamodel, redshift_max, duration,plot_show=True):
    """
    Plot observed magnitude vs. redshift for simulated SNIa events.

    Args:
        sniadata (pandas.DataFrame): Data of simulated SNIa events.
        redshift_max (float): Maximum redshift for the plot.
    """
    
    fig = sniamodel.show_scatter("z", "US-Band", ckey="c")
    ax = fig.axes[0]

    # Add annotations and plot formatting
    plt.plot([-100, 100], [22.5, 22.5], color="black")
    ax.set_ylabel("Observed magnitude in AB and USAT-Band")
    ax.set_xlabel("Redshift")
    ax.set_xlim(0, redshift_max + 0.01)
    ax.set_ylim(14, 40)
    ax.text(0.153, 31, "Reddening c", fontsize=10, color="black", rotation=90)
    ax.text(0.09, 21.7, "Limiting magnitude", fontsize=8, color="black")
    plt.gca().invert_yaxis()
    fig.suptitle(str(duration)+' days simulation of SNIa events at the full sky in Ultrasat-Band', y=1)

    # Save and display the plotbase_filename = "parameter_comparison_table_"+str(len(comparison_table))+"_entries.csv"
    base_filename="All_simulations"
    unique_filename = generate_unique_filename("Results", base_filename)
    plt.savefig(unique_filename, dpi=600)
    if plot_show==True:
        plt.show()



def convert_drawn_data_to_instance(template_name, sniadata, sniamodel):
    """
    Convert simulated SNIa data to instance.

    Args:
        template_name (str): Name of the SNIa template to use (e.g., "QinanSalt3").
        sniadata (pandas.DataFrame): Simulated SNIa data.

    Returns:
        model instance of the drawn data
    """
    

    sniainstance = sniamodel.from_data(sniadata, template=template_name)
    return sniainstance

#_______________________________________
#DUST

def simulate_snia_events_dust(template_name, redshift_max, start_time, duration,dust_extinction, magnitude_limit):
    """
    Simulate SNIa events for a given template and redshift range.

    Args:
        template_name (str): Name of the SNIa template to use (e.g., "QinanSalt3").
        redshift_max (float): Maximum redshift for the simulation.
        start_time (float): Start time for the simulation (e.g., MJD).
        duration (float): Duration of the simulation in days.
        magnitude_limit (float): Magnitude limit for filtering events.

    Returns:
        pandas.DataFrame: Simulated SNIa events with additional magnitude data.
    """
    # Initialize SNIa simulation object. Add dust parameters for drawing and set reddening parameter c to 0.

    SNeIa._MODEL = SNeIa._MODEL.copy()
    SNeIa._MODEL["hostr_v"] = {"func": np.random.uniform,"kwargs": {"low":3.1, "high":3.1} }

    SNeIa._MODEL["hostebv"] = {
        "func": np.random.exponential,
        "kwargs": {"scale": 0.05}  # Is this realistic?
    }
    SNeIa._MODEL["c"] = {"func": np.random.uniform,"kwargs": {"low":0, "high":0} }


    snia = SNeIa()
    snia.set_template(template_name)
    if dust_extinction:
        snia.add_effect(effect=effects.mw_extinction)
    #Link the drawn dust values to the dust effect.

    host_effect = Effect(effect=sncosmo.CCM89Dust(), name="host", frame="rest")

    # Define how to sample the parameters for the effect
    hostrelation = {
        "hostebv": {
            "func": np.random.uniform,
            "kwargs": {"low": 0, "high": 0.3}
        },
        "hostr_v": {
            "func": np.random.normal,
            "kwargs": {"loc": 3.1, "scale": 0.1}
        }
    }

    # Add the effect to the target with the correct model format
    snia.add_effect(effect=host_effect, model=hostrelation)

    #snia.set_template(template_name)
    #print(snia.get_template)
    
    # Draw supernova events
    num_events = int(snia.get_rate(redshift_max) / (365/duration))

    sniadata = snia.draw(size=num_events, zmax=redshift_max, zmin=0,
                         tstart=start_time, tstop=start_time + duration, inplace=True)

    # Calculate observed magnitudes in the specified band
    target_magnitudes = []
    for j in range(len(sniadata)):
        mag = snia.get_target_mag(j, band="ultrasat_band_0.00", phase=0, magsys="ab", restframe=True)
        target_magnitudes.append(mag)
    sniadata["US-Band"] = target_magnitudes

    # Apply magnitude filter
    sniadata = sniadata[sniadata["US-Band"] <= magnitude_limit]
    sniadata = sniadata.reset_index(drop=True)
    
    # Take the model for future usage
    sniamodel=snia

    return sniadata,sniamodel

def simulate_snia_events_dust_varRV(template_name, redshift_max, start_time, duration,dust_extinction, magnitude_limit):
    """
    Simulate SNIa events for a given template and redshift range.

    Args:
        template_name (str): Name of the SNIa template to use (e.g., "QinanSalt3").
        redshift_max (float): Maximum redshift for the simulation.
        start_time (float): Start time for the simulation (e.g., MJD).
        duration (float): Duration of the simulation in days.
        magnitude_limit (float): Magnitude limit for filtering events.

    Returns:
        pandas.DataFrame: Simulated SNIa events with additional magnitude data.
    """
    # Initialize SNIa simulation object. Add dust parameters for drawing and set reddening parameter c to 0.

    SNeIa._MODEL = SNeIa._MODEL.copy()
    SNeIa._MODEL["hostr_v"] = {
        "func": np.random.normal,
        "kwargs": {"loc": 3.1, "scale": 0.1}
    }
    SNeIa._MODEL["hostebv"] = {
        "func": np.random.exponential,
        "kwargs": {"scale": 0.05}  # Is this realistic?
    }
    SNeIa._MODEL["c"] = {"func": np.random.uniform,"kwargs": {"low":0, "high":0} }


    snia = SNeIa()
    snia.set_template(template_name)
    if dust_extinction:
        snia.add_effect(effect=effects.mw_extinction)
    #Link the drawn dust values to the dust effect.

    host_effect = Effect(effect=sncosmo.CCM89Dust(), name="host", frame="rest")

    # Define how to sample the parameters for the effect
    hostrelation = {
        "hostebv": {
            "func": np.random.uniform,
            "kwargs": {"low": 0, "high": 0.3}
        },
        "hostr_v": {
            "func": np.random.normal,
            "kwargs": {"loc": 3.1, "scale": 0.1}
        }
    }

    # Add the effect to the target with the correct model format
    snia.add_effect(effect=host_effect, model=hostrelation)

    #snia.set_template(template_name)
    #print(snia.get_template)
    
    # Draw supernova events
    num_events = int(snia.get_rate(redshift_max) / (365/duration))

    sniadata = snia.draw(size=num_events, zmax=redshift_max, zmin=0,
                         tstart=start_time, tstop=start_time + duration, inplace=True)

    # Calculate observed magnitudes in the specified band
    target_magnitudes = []
    for j in range(len(sniadata)):
        mag = snia.get_target_mag(j, band="ultrasat_band_0.00", phase=0, magsys="ab", restframe=True)
        target_magnitudes.append(mag)
    sniadata["US-Band"] = target_magnitudes

    # Apply magnitude filter
    sniadata = sniadata[sniadata["US-Band"] <= magnitude_limit]
    sniadata = sniadata.reset_index(drop=True)
    
    # Take the model for future usage
    sniamodel=snia

    return sniadata,sniamodel


def plot_snia_data_dust(sniamodel, redshift_max, duration, plot_show=True):
    """
    Plot observed magnitude vs. redshift for simulated SNIa events. With R_V as color parameter.

    Args:
        sniadata (pandas.DataFrame): Data of simulated SNIa events.
        redshift_max (float): Maximum redshift for the plot.
    """
    
    fig = sniamodel.show_scatter("z", "US-Band", ckey="hostr_v")
    ax = fig.axes[0]

    # Add annotations and plot formatting
    plt.plot([-100, 100], [22.5, 22.5], color="black")
    ax.set_ylabel("Observed magnitude in AB and USAT-Band")
    ax.set_xlabel("Redshift")
    ax.set_xlim(0, redshift_max + 0.01)
    ax.set_ylim(14, 40)
    ax.text(0.153, 31, "Dust parameter R_v", fontsize=10, color="black", rotation=90)
    ax.text(0.09, 21.7, "Limiting magnitude", fontsize=8, color="black")
    plt.gca().invert_yaxis()
    fig.suptitle(str(duration)+' days simulation of SNIa events at the full sky in Ultrasat-Band', y=1)

    # Save and display the plot
    base_filename="All_simulations_dust"
    unique_filename = generate_unique_filename("Results", base_filename)
    print(unique_filename)
    plt.savefig(unique_filename, dpi=600)
    if plot_show==True:
        plt.show()