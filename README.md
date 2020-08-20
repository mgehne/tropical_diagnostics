# tropical-diagnostics
Python scripts for tropical diagnostics of NWP forecasts.

The diagnostics are meant to be applied to forecast data and the example scripts show how to apply the diagnostics at different lead times.

# diagnostics
Contains the functions and modules necessary to compute the various diagnostics. The main diagnostics included are: Hovmoeller diagrams, space-time spectra, convectively coupled equatorial wave (CCEW) activity and skill, and vertical coherence of CCEWs.

# utils
Contains functions and modules shared by multiple diagnostics. That includes reading data, saving netcdf files and AJM source code.

# examples
Scripts containing example use cases. These scripts read in data, compute diagnostics and plot the results. This is a good place to start when first using the diagnostics. 
The user will need to supply their own data and edit these examples to get them to work.

