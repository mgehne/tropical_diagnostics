"""
This script reads in data from a user specified file and filters it for
a specified region in wavenumber - frequency space.
User must set filename and path for input and output files, define start
and end time of the data, define the region to filter for, set the latitude
range to filter and specify the number of observations per day of the input
data.
This can be very slow if the data is high spatial and temporal resolution.
"""
import numpy as np
import xarray as xr
from diagnostics import spacetime as st
import time as systime
import sys


# file and pathname
filein = 'precip.trmm.8x.v7a.fillmiss.comp.1998-201806'
pathin = '/data/mgehne/Precip/'
pathout = '/data/mgehne/Precip/MetricsObs/CCEWfilter/'

# number of obs per day
spd = 8
datestrt = "1998-01-01"
datelast = "2017-12-31"

# set the wave to filter for
waveName = 'Kelvin'

# filename for filtered datra
fileout = filein+'.'+waveName

# parameters for filtering the data
latMin = -20
latMax = 20

# values for filtering regions
if waveName == "Kelvin":
    # Filter for Kelvin band
    tMin = 2.5
    tMax = 20
    kMin = 1
    kMax = 14
    hMin = 8
    hMax = 90
elif waveName == "MRG":
    # Filter for 2-6 day bandpass
    tMin = 2
    tMax = 6
    kMin = -250
    kMax =  250
    hMin = -9999
    hMax = -9999
elif waveName == "IG1":
    # Filter for WIGs
    tMin = 1.2
    tMax = 2.6
    kMin = -15
    kMax = -1
    hMin = 12
    hMax = 90
elif waveName == "ER":
    tMin = 10
    tMax = 40
    kMin = -10
    kMax = -1
    hMin = 8
    hMax = 90
elif waveName == "MJO":
    # Filter for 30-96 day eastward
    tMin = 30
    tMax = 96
    kMin = 0
    kMax = 250
    hMin = -9999
    hMax = -9999
else:
    print("Please define a region to filter for.")
    sys.exit()


# read data from file
print("open data set...")
ds = xr.open_dataset(pathin+filein+".nc")
data = ds['precip'].sel(lat=slice(latMin, latMax),time=slice(datestrt, datelast))
lat = ds['lat'].sel(lat=slice(latMin, latMax))
lon = ds['lon']
time = ds['time'].sel(time=slice(datestrt, datelast))
ds.close()
print("done. size of data array:")
print(data.shape)

# filter each latitude
datafilt = xr.DataArray(np.zeros(data.shape), dims=['time', 'lat', 'lon'])
print("Filtering....")
for ll in range(len(lat)):
    tstrt = systime.process_time()
    print("latitude "+str(lat[ll].values))
    print("filtering current latitude")
    datafilt2d = st.kf_filter(np.squeeze(data[:, ll, :].values), spd, tMin, tMax, kMin, kMax, hMin, hMax, waveName)
    print("write data for current latitude to array")
    datafilt[:, ll, :] = datafilt2d
    print(systime.process_time()-tstrt, 'seconds')
print("Done!")
print(datafilt.min(), datafilt.max())

# save filtered data to file
print("save filtered data to file")
ds = xr.Dataset({'precip': datafilt}, {'time':time, 'lat':lat, 'lon':lon})
ds.to_netcdf(pathout+fileout+".nc")
ds.close()