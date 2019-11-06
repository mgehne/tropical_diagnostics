import xarray as xr
"""
local scripts, if loading from a different directory include that with a '.' between
directory name and script name
"""
from tropical_diagnostics.diagnostics.spacetime import mjo_cross
from tropical_diagnostics.diagnostics.spacetime import get_symmasymm
from tropical_diagnostics.utils.save_netcdf import save_Spectra

"""
Set parameters for the spectra calculation.
spd:  Number of observations per day.
nperseg:  Number of data points per segment.
segOverlap:  How many data points of overlap between segments. If negative there is overlap
             If positive skip that number of values between segments.
Symmetry:    Can be "symm", "asymm" for symmetric or anti-symmetric across the equator. "latband" 
             if the spectra averaged across the entire latitude band are requested.
latMin:      Minimum latitude to include.
latMax:      Maximum latitude to include.
"""
spd = 2
nperseg = 90*spd
segOverLap = -30*spd
Symmetry = "asymm"
latMin = -15.
latMax =  15.

print("reading data from file:")
""" 
Read in data here. Example:
ds = xr.open_dataset('/data/mgehne/ERAI/MetricsObs/precip.erai.sfc.1p5.2x.1979-2016.nc')
x = ds.u
x = x.sel(lat=slice(latMin,latMax)) 
x = x.squeeze()
"""
ds = xr.open_dataset('/data/mgehne/ERAI/MetricsObs/precip.erai.sfc.1p5.2x.1979-2016.nc')
x = ds.precip
x = x.sel(lat=slice(latMin,latMax))
x = x.squeeze()
latA = ds.lat.sel(lat=slice(latMin,latMax))
ds.close()

ds = xr.open_dataset('/data/mgehne/ERAI/MetricsObs/div.erai.850.1p5.2x.1979-2016.nc')
y = ds.div
y = y.sel(lat=slice(latMin,latMax))
y = y.squeeze()
latB = ds.lat.sel(lat=slice(latMin,latMax))

if any(latA-latB)!=0:
  print("Latitudes must be the same for both variables! Check latitude ordering.")

print("get symmetric/anti-symmetric components:")
if Symmetry=="symm" or Symmetry=="asymm":
    X = get_symmasymm(x,latA,Symmetry)
    Y = get_symmasymm(y,latB,Symmetry)
else:
    X = x
    Y = y
    
print("compute cross-spectrum:")
"""
The output from mjo_cross includes:
STC = [8,nfreq,nwave]
STC[0,:,:] : Power spectrum of x
STC[1,:,:] : Power spectrum of y
STC[2,:,:] : Co-spectrum of x and y
STC[3,:,:] : Quadrature-spectrum of x and y
STC[4,:,:] : Coherence-squared spectrum of x and y
STC[5,:,:] : Phase spectrum of x and y
STC[6,:,:] : Phase angle v1
STC[7,:,:] : Phase angle v2
freq 
wave
number_of_segments
dof
prob
prob_coh2
"""
result = mjo_cross(X, Y, nperseg, segOverLap)  
STC = result['STC'] #, freq, wave, number_of_segments, dof, prob, prob_coh2

freq = result['freq']
freq = freq*spd
wnum = result['wave']

# save spectra in netcdf file

fileout = 'SpaceTimeSpectra_'+Symmetry+'_'+str(spd)+'spd'
pathout = '../data/'
print('saving spectra to file: '+pathout+fileout+'.nc')
save_Spectra(STC,freq,wnum,fileout,pathout)
