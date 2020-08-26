"""
Example script to compute vertical coherence profiles.

This script assumes that the filtered precipitation file already exists. If not, the user needs to
run filter_CCEW.py to generate the filtered data. The user needs to change the input paths and
filenames and the output location. Variable names depend on the input data and need to be specified
by the user as well.
"""
from tropical_diagnostics.diagnostics import vertical_coherence as vc
import xarray as xr
import numpy as np

var1 = "precip"  # variable name of data in the precipitation file
var2 = "shum"    # variable name of data in the second file

source1 = "TRMM"  # ERAI, ERA5, TRMM
source2 = "ERA5"

wave1 = "MJO"  # "kelvin", "MRG", "ER", "MJO", "IG0"

RES = "2p5"  # spatial resolution of the data
spd = 1  # data is spd x daily

# var2 is read at all these levels
#level2 = [1000,975,950,925,900,875,850,825,800,775,750,700,650,600,550,500,
#          450,400,350,300,250,225,200,175,150,125,100]
level2 = [1000,200]

# first and last date format: yyyymmddhh
datemin = '2007-01-01'
datemax = '2010-12-31'
yearmin = datemin[0:4]
yearmax = datemax[0:4]

# significance level for the coherence plots
sigstr = 99.
siglev = sigstr/100

# latBound
latN = 20
latS = -latN

# Wheeler - Kiladis[WK] temporal window length(days)
nDayWin = 128 * spd
nDaySkip = -32 * spd

# input file names
filebase = 'precip.trmm.'+str(spd)+'x.'+RES+'.v7a.fillmiss.comp.1998-201806'
pathin = '/data/mgehne/Precip/MetricsObs/CCEWfilter/'
pathout = '/data/mgehne/VerticalCoherence/'
plotpath = "~/Projects/Diagnostics/Plots/VerticalCoherence/"
outfile = "CoherenceVertical_python_"+RES+"_"+str(spd)+"x_"+source1+var1+wave1+"_"+source2+var2+"_"\
          + str(datemin)+"-"+str(datemax)+"_"+str(latS)+"-"+str(latN)+"_sigMask"
outfileSpectra = "CoherenceVertical_SpaceTime_python_"+RES+"_"+str(spd)+"x_"+source1+var1+wave1+"_"+source2+var2+"_"\
          + str(datemin)+"-"+str(datemax)+"_"+str(latS)+"-"+str(latN)+"_sigMask"


# read data1
print('reading data1')
ds = xr.open_dataset(pathin+filebase+"."+wave1+".nc")
data1 = ds[var1].sel(lat=slice(latS,latN), time=slice(datemin,datemax))
ds.close()

# read data2
print('reading data2')
ds = xr.open_dataset('/data/mgehne/era5/shum.2p5.daily.2007-2010.nc')
data2 = ds[var2].sel(lat=slice(latN,latS), time=slice(datemin,datemax), level=level2)
ds.close()

print(data1.shape)
print(data2.shape)

# put this next part into a function
CrossAvg, CrossMask, CrossMat = vc.vertical_coherence_comp(data1, data2, level2, nDayWin, nDaySkip, spd, siglev)
print(CrossAvg)

# save data to file
#ds = xr.Dataset({'CrossAvg': CrossAvg}, {'level': level2, 'cross': np.arange(0, 16, 1)})
#ds.to_netcdf(pathout+outfile+".nc")
#ds.close()

# plot vertical coherence profile
