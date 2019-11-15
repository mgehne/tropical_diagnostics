import numpy as np
import xarray as xr
import sys

sys.path.append('../../')
"""
local scripts, if loading from a different directory include that with a '.' between
directory name and script name
"""
from tropical_diagnostics.diagnostics.hovmoeller_plotly import hovmoeller
from tropical_diagnostics.diagnostics.hovmoeller_calc import lat_avg
from tropical_diagnostics.diagnostics.hovmoeller_calc import pattern_corr

plotpath = '../plots/'

"""
Parameters to set for the Hovmoeller diagrams.
"""
datestrt = '2016-01-01'  # plot start date, format: yyyy-mm-dd
datelast = '2016-03-31'  # plot end date, format: yyyy-mm-dd
#contourmin = 0.2  # contour minimum
#contourmax = 1.2  # contour maximum
#contourspace = 0.2  # contour spacing
latMax = 5.  # maximum latitude for the average
latMin = -5.  # minimum latitude for the average

print("reading data from file:")
spd = 2  # number of obs per day
source = "ERAI"  # data source
var = "precip"  # variable to plot
lev = ""   # level
ds = xr.open_dataset('/data/mgehne/ERAI/MetricsObs/precip.erai.sfc.1p5.2x.1979-2016.nc')
A = ds.precip
lonA = ds.lon
print("extracting time period:")
A = A.sel(time=slice(datestrt, datelast))
timeA = ds.time.sel(time=slice(datestrt, datelast))
ds.close()

print("average over latitude band:")
A = A * 1000 / 4
A.attrs['units'] = 'mm/day'
A = lat_avg(A, latmin=latMin, latmax=latMax)

# print("plot hovmoeller diagram:")
# hovmoeller(A, lonA, timeA, datestrt, datelast, plotpath, latMin, latMax, spd, source, var, lev,
#           contourmin, contourmax, contourspace)
# hovmoeller(A, lonA, timeA, datestrt, datelast, plotpath, latMin, latMax, spd, source, var, lev)

var = "uwnd"  # variable to plot
lev = "850"   # level
ds = xr.open_dataset('/data/mgehne/ERAI/MetricsObs/uwnd.erai.850.1p5.2x.1979-2016.nc')
B = ds.uwnd
print("extracting time period:")
B = B.sel(time=slice(datestrt, datelast))
ds.close()

B = lat_avg(B, latmin=latMin, latmax=latMax)


corr = pattern_corr(A, B)
print(corr)
