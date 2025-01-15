# Simulation of SNeIa with Skysurvey in the ULTRASAT Survey

This is a set of Python script for deriving an estimated rate of usable SNeIa observations by ULTRASAT. It is not optimized for public use yet, even though the important parts are included.

## IMPORTANT:
This repository contains work in progress and unpublished data. Please do not distribute or share it publicly. 
Particularly, the content of the `Data` folder is mostly not published yet.


## Explanations:

### ultrasat_simulation_*.py scripts
The `ultrasat_simulation_*.py` files in the main folder are the runnable scripts. They call the modules in the `Modules` folder and the `config.yaml`, which contains the most important configurations.

- The most important script is `_lightcurves.py`, as it produces the light curves and saves them in the `Lightcurve` folder. It also creates some overview plots.  
  During the process, it generates a "survey". This takes quite a long time, but the result is saved in the `Cache` folder because it can be reused when the same survey configuration is applied.
- `_rates.py` and `_earlypeaks.py` use the light curves from the `Lightcurve` folder to calculate the desired rates. These are partly tailored towards the SALT model used for SNeIa. At the beginning of these scripts, you can define the folder in `HCorLC`.
- `_plotting.py` creates overview plots with different sigma clipping.
- `_lightcurves_Blackbody.py` is used to create blackbody data to check with the [SNR Calculator](https://www.weizmann.ac.il/ultrasat/for-scientists/snr-calculator).
- `_fitting.py` and `_fitting_Blackbody.py` are scripts that produce light curves and fit them using the implemented template. These scripts are outdated and might not work immediately.

### config.yaml
This text file contains all the configurable parameters that may be defined before runs. In many cases, the parameters are self-explanatory, but a look into the scripts may be necessary to fully understand how they work.

### Modules
The `Modules` folder contains the scripts that are called by the main scripts. Sometimes, alternative versions of some functions are defined in the lower sections of these files for special cases. The most important ones are generally in the first sections.

- `models.py` creates a SALT3 template.
- `filters.py` defines the ULTRASAT wavebands.
- `simulation.py` generates the targets and applies dust extinction to them.
- `survey_plan.py` creates the survey plan. It also computes the radial offsets of the Healpix IDs and assigns the relevant parameters accordingly.
- `lightcurves.py` combines these components to simulate observations and performs some post-processing.
- `plotting.py` generates an overview plot of all observed objects.
- `curve_fit.py` fits the generated light curves.
- `ZTF_survey.py` is used in the fitting process to incorporate additional bands from ZTF logs.

### Data
The `Data` folder contains text files from external sources, mostly from Yossi Shvartzvald regarding ULTRASAT specifics, as well as the UV-improved SALT3 template by Qinan Wang.
