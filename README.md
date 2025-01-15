# Simulation of transients with Skysurvey in the ULTRASAT Survey

This is a set of Python script for deriving an estimated rate of usable SNeIa observations by ULTRASAT. The goal is to make it usable for any transient. It is not optimized for public use yet, even though the important parts are included.

## IMPORTANT:
This repository contains work in progress and unpublished data. Please do not distribute or share it publicly. 
Particularly, the content of the `Data` folder is mostly not published yet.


## Explanations:

### ultrasat_simulation_*.py scripts
The `ultrasat_simulation_*.py` files in the main folder are the runnable scripts. They call the modules in the `Modules` folder and the `config.yaml`, which contains the most important configurations.

- The most important script is `lightcurves.py`, as it produces the light curves and saves them in the `Lightcurve` folder. It also creates some overview plots.  
  During the process, it generates a "survey". This takes quite a long time, but the result is saved in the `Cache` folder because it can be reused when the same survey configuration is applied.
- `rates.py` and `earlypeaks.py` use the light curves from the `Lightcurve` folder to calculate the desired rates. These are partly tailored towards the SALT model used for SNeIa. At the beginning of these scripts, you can define the folder in `HCorLC`.
- `plotting.py` creates overview plots with different sigma clipping.
- `lightcurves_Blackbody.py` is used to create blackbody data to check with the [SNR Calculator](https://www.weizmann.ac.il/ultrasat/for-scientists/snr-calculator).
- `fitting.py` and `fitting_Blackbody.py` are scripts that produce light curves and fit them using the implemented template. These scripts are outdated and might not work immediately.

### config.yaml
This text file contains all the configurable parameters that may be defined before runs. In many cases, the parameters are self-explanatory, but a look into the scripts may be necessary to fully understand how they work.

### Modules
The `Modules` folder contains the scripts that are called by the main scripts. Sometimes, alternative versions of some functions are defined in the lower sections of these files for special cases. The most important ones are generally in the first sections.

- `templates.py` creates a SALT3 template.
- `filters.py` defines the ULTRASAT wavebands.
- `simulation.py` generates the targets and applies dust extinction to them.
- `survey_plan.py` creates the survey plan. It also computes the radial offsets of the Healpix IDs and assigns the relevant parameters accordingly.
- `lightcurves.py` combines these components to simulate observations and performs some post-processing.
- `plotting.py` generates an overview plot of all observed objects.
- `curve_fit.py` fits the generated light curves.
- `ZTF_survey.py` is used in the fitting process to incorporate additional bands from ZTF logs.

### Data
The `Data` folder contains text files from external sources, mostly from Yossi Shvartzvald regarding ULTRASAT specifics, as well as the UV-improved SALT3 template by Qinan Wang.

## Instructions:
To adapt the simulation for **another type of transients**, I would expect the following steps to run the code. Note that the current code does not fully support all of these steps, so modifications might be required:

0. **Read the documentation** about [transients](https://skysurvey.readthedocs.io/en/latest/quickstart/quickstart_target.html) in the Skysurvey wiki.
1. **Import the template** of the transient you want to simulate to the `Data` folder. Update the path in `config.yaml` under `template_directory`. Optionally, modify names in `templates.py` and `config.yaml` as needed.
2. **Update the event rates**: Incorporate the expected yearly rates over the full sky of these transients by updating the `num_events` line under `def simulate_snia_events` in the module `simulation.py`.
3. **Create a transient model**: Understand how transient models work in Skysurvey and create a new model. The model should depend on the template you are using and realistically reflect the distribution of template parameters in nature. As skysurvey comes only with an SNIa model, this step requires sustantial changes in the code, but skysurvey provides a possibility to integrate other transient models.
4. **Select a limiting magnitude source**:
   - Compare your template with the available sources to determine which spectrum is closest.
   - For blackbodies, there is existing code in `simulation.py` and `ultrasat_simulation_lightcurves_Blackbody.py`.
   - For stellar templates, refer to [this website](http://cdsarc.u-strasbg.fr/viz-bin/ftp-index?J/PASP/110/863).
   - Set the according number for `source_number` in `config.yaml`.
5. **Configure and run**:
   - Decide on all the other parameters in `config.yaml`.
   - Run `ultrasat_simulation_lightcurves.py`.
   - Check the `Lightcurves` folder to verify if the simulation was successful.





