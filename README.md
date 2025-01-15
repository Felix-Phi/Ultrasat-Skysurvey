# Simulation of SNeIa with Skysurvey in the ULTRASAT Survey

This is my Python script for deriving an estimated rate of observable SNeIa. It is not optimized for public use yet and one will probably run into some problems when it is used right away, even thought the important parts are there.

## Explainations:

#### ultrasat_simulation_*.py scripts
The ultrasat_simulation_*.py-files in the main folder are the runable scripts. They call the modules in the Modules-folder and the config.yaml, that contains the most important configurations.
- The most important one is "_lightcurves.py" as it produces the lightcurves and puts them into the folder named "Lightcurve". It also creates some overview plots. 
In the process it creates a "survey". This takes rather long and the result is saved into the Cache-Folder, because it is reusable when one uses the same configuration. 
- "_rates.py" and "_earlypeaks.py" use these lightcurves in the "Lightcurve"-folder and create the desired rates. These are partly tailored towards the SALT Model used for SNeIa. In the beginning you can define the used folder in "HCorLC"
- "_plotting.py" create overview plots with different clipping.
- "_lightcurves_Blackbody.py" is a script that is used for creating blackbody data to compare it to the SNR-Calculator (https://www.weizmann.ac.il/ultrasat/for-scientists/snr-calculator).
- "_fitting.py" and "_fitting_Blackbody.py" are autonomous scripts that fit the produced that using the implemented template. These are outdated and might not work immediatly.

#### config.yaml
In this text-file all the open configurations are set, that might change from run to run. In many cases they are self explanatory but maybe one has to look into the scripts to understand what they actually define.

#### Modules
In the "Modules"-folder are the called scripts. Sometimes there are alternative versions to some definitions in the lower section that are used for special cases, so the most interesting parts are in the first part.

- "models.py" creates a SALT3 template.
- "filters.py" creates the ULTRASAT wavebands.
- "simulation.py" draws the targets and applies the dust extinction on them.
- "survey_plan.py" creates the survey plan. It also computes the radial offsets of the Healpix-IDs and assigns the relevant parameters accordingly.
- "lightcurves.py" combines these two to get the simulated observations. Also some post-processing.
- "plotting.py" creates a plot that shows an overview over all the observed objects.
- "curve_fit.py" fits the created lightcurves.
- "ZTF_survey.py" is used in the fitting to add other bands out of ZTF logs.

#### Data
In the Data-folder are text files from external sources. It is mostly from Yossi Shvartzvald regarding ULTRASAT specifics but also the UV-improved SALT3 template by Qinan Wang.
