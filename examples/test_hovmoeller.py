import numpy as np
import xarray as xr
import sys
sys.path.append('../../')
"""
local scripts, if loading from a different directory include that with a '.' between
directory name and script name
"""
from tropical_diagnostics.diagnostics.hovmoeller import hovmoeller

plotpath = '../plots/'

"""
Parameters to set for the Hovmoeller diagrams.
"""
spd = 2               # number of obs per day
source = "ERAI"      # data source
var1 = "precip"       # variable to plot
datestrt = 2016010100 # plot start date, format: yyyymmddhh 
datelast = 2016033100 # plot end date, format: yyyymmddhh
contourmin = 0.001     # contour minimum
contourmax = 0.01      # contour maximum
contourspace = 0.001    # contour spacing
latMax =  5.            # maximum latitude for the average 
latMin = -5.            # minimum latitude for the average 

print("reading data from file:")
ds = xr.open_dataset('/data/mgehne/ERAI/MetricsObs/precip.erai.sfc.1p5.2x.1979-2016.nc')
A = ds.precip
lonA = ds.lon
print("extracting latitude bands:")
A = A.sel(lat=slice(latMin,latMax))
A = A.squeeze()
latA = ds.lat.sel(lat=slice(latMin,latMax))
A = A.sel(time=slice('2016-01-01','2016-12-31'))
A = A.squeeze()
timeA = ds.time.sel(time=slice('2016-01-01','2016-12-31'))
ds.close()

print(timeA)
    
print("average over latitude band:")
units = A.attrs['units']
A = A.mean(dim='lat')
A = A*1000/4
A.attrs['units'] = 'mm/day'

A.min(), A.max()

print("plot hovmoeller diagram:")
hovmoeller(A,lonA,timeA,datestrt,datelast,spd,source,var1,plotpath,latMin,latMax)
#hovmoeller(A,lonA,timeA,datestrt,datelast,spd,source,var1,plotpath,latMin,latMax,contourmin,contourmax,contourspace)
