# tropical-diagnostics
Tropical Diagnostics Toolbox: Python scripts for tropical diagnostics of NWP forecasts.

The diagnostics are meant to be applied to gridded forecast data and example scripts are provided to show how
to apply the diagnostics at different lead times.

Required model output is primarily precipitation. This is enough to compute Hovmoeller diagrams and
compare to observations and to project onto the convectively coupled equatorial wave (CCEW) EOFs to
analyze CCEW activity and skill in model forecasts.

For the moisture convection coupling analysis vertical level data of specific humidity and temperature are needed,
in addition to surface pressure and a land-sea mask. The vertical coherence diagnostic also uses data at vertical
levels: temperature, specific humidity, divergence, zonal and meridional winds. However, it is up to the user to decide
which variables are most useful to the user's specific application.

## tropical_diagnostics
Contains the functions and modules necessary to compute the various diagnostics. The main diagnostics
included are:

### Hovmoeller diagrams
Functions to compute hovmoeller latitudinal averages and pattern correlation are included in
**hovmoeller_calc.py**. Plotting routines are included in **hovmoeller_plotly.py**.

### Space-time spectra
Functions for computing 2D Fourier transforms and 2D power and cross-spectra are included in **spacetime.py**.
To plot the spectra **spacetime_plot.py** uses pyngl, which is based on NCL and provides similar control
over plotting resources.

### CCEW activity and skill
Functions to project precipitation (either from model output or observations) onto CCEW EOF patterns and
compute wave activity and a CCEW skill score are included in **CCEWactivity.py**. Also included are routines
to plot the activity and the skill compared to observations.

### Vertical coherence of CCEWs
These functions compute the coherence between two data sets at multiple vertical levels. The assumption is that
the first data set is filtered precipitation (filtered for a CCEW or the MJO) and the second data set is a multi-
level dynamical field, either from reanalysis or model output.
Once coherence between all variables and filtered precipitation has been computed and saved the plotting routines
to plot vertical profiles of coherence and phase angles can be used. The plotting is done using plotly and the kaleido
engine. Example scripts are provided for both the computational (**vertical_coherence.py**) and plotting
(**vertical_coherence_plot.py**) part.

### Moisture - Convection coupling
These functions contain code to compute column saturation fraction (CSF) from reanalysis or model output and bin CSF
and precipitation against each other (**moisture_convection_coupling.py**). Routines for plotting the circulation in
preciptiation - CSF space and the distributions are contained in **moisture_convection_coupling_plot.py**. The example
script **precip_csf_coupling.py** computes CSF from model output, bins TRMM precipitation against model CSF and plots
the results. **B_L_analysis.py** shows how to run the B_L analysis scripts and plots results.

## examples
Scripts containing example use cases. These scripts read in data, compute diagnostics and plot the results.
This is a good place to start when first using the diagnostics.
The user will need to supply their own data and edit these examples to get them to work.

## installation
Download or clone the source code from github. Make sure all the required packages are available. If working
on a machine that requires python environments to install packages create a conda environment first:

MyEnv = your chosen name of the environment

`conda create --name MyEnv`

Install the required packages:

`conda install -n MyEnv scipy`

`conda install -n MyEnv xarray`

`conda install -n MyEnv numpy`

`conda install -n MyEnv netCDF4`

`conda install -n MyEnv pyngl`

Activate the conda environment:
`conda activate MyEnv`

Change directory to top level package directory tropical_diagnostics and install package using pip.

`cd tropical_diagnostics/`
`pip install ./`

To run scripts from the examples directory copy the script into a directory where you would like to run. Make
sure to activate the conda environment with `conda activate MyEnv`. Adjust all path and filenames if necessary.
