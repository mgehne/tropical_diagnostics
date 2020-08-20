# tropical-diagnostics
Python scripts for tropical diagnostics of NWP forecasts.

The diagnostics are meant to be applied to gridded forecast data and example scripts are provided to show how
to apply the diagnostics at different lead times.

Required model output is primarily precipitation. This is enough to compute Hovmoeller diagrams and
compare to observations and to project onto the convectively coupled equatorial wave (CCEW) EOFs to
analyze CCEW activity and skill in model forecasts.

## diagnostics
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
    Needs to still be included.

## utils
Contains functions and modules shared by multiple diagnostics. That includes reading data, saving netcdf
files and AJM source code.

## examples
Scripts containing example use cases. These scripts read in data, compute diagnostics and plot the results.
This is a good place to start when first using the diagnostics.
The user will need to supply their own data and edit these examples to get them to work.

