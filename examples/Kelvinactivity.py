import numpy as np
import xarray as xr
import sys

sys.path.append('../../')
"""
local scripts, if loading from a different directory include that with a '.' between
directory name and script name
"""
from tropical_diagnostics.diagnostics import CCEWactivity

plotpath = '../plots/'
eofpath  = '../data/EOF/'

"""
Parameters to set for plotting Kelvin activity index.
"""
wave = 'Kelvin'
datestrt = '2016-01-01'
datelast = '2016-03-31'

print("reading ERAI data from file:")
ds = xr.open_dataset('/data/mgehne/ERAI/MetricsObs/precip.erai.sfc.1p5.2x.1979-2016.nc')
A = ds.precip
print("extracting time period:")
A = A.sel(time=slice(datestrt, datelast))
A = A.squeeze()
timeA = ds.time.sel(time=slice(datestrt, datelast))
ds.close()
A = A * 1000 / 4
A.attrs['units'] = 'mm/day'

print("reading PERSIANN data from file:")
ds = xr.open_dataset('/data/mgehne/Precip/MetricsObs/persiann_cdr_1p5_fillmiss8314_1983-2016.nc')
B = ds.precip
print("extracting time period:")
B = B.sel(time=slice(datestrt, datelast))
B = B.squeeze()
timeB = ds.time.sel(time=slice(datestrt, datelast))
ds.close()

print("project data onto wave EOFs")
#waveactA = CCEWactivity.waveact(A, wave, eofpath)
#print(waveactA)
waveactB = CCEWactivity.waveact(B, wave, eofpath)
print(waveactB)

print("plot activity")
