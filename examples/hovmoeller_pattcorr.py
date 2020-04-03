import numpy as np
import xarray as xr
import sys

sys.path.append('../../')
"""
local scripts, if loading from a different directory include that with a '.' between
directory name and script name
"""
from tropical_diagnostics.diagnostics.hovmoeller_plotly import plot_pattcorr
from tropical_diagnostics.diagnostics.hovmoeller_calc import lat_avg
from tropical_diagnostics.diagnostics.hovmoeller_calc import pattern_corr

plotpath = '../plots/'

"""
Parameters to set for the Hovmoeller diagrams.
"""
datestrt = '2016-01-11'  # plot start date, format: yyyy-mm-dd
datelast = '2016-03-31'  # plot end date, format: yyyy-mm-dd
latMax = 5.  # maximum latitude for the average
latMin = -5.  # minimum latitude for the average

print("reading observed precipitation data from file:")
spd = 1
ds = xr.open_dataset('/data/mgehne/Precip/MetricsObs/precip.trmm.'+str(spd)+'x.1p0.v7a.fillmiss.comp.1998-2016.nc')
B = ds.precip
print("extracting time period:")
B = B.sel(time=slice(datestrt, datelast))
B = B.squeeze()
timeB = ds.time.sel(time=slice(datestrt, datelast))
ds.close()
B.attrs['units'] = 'mm/d'
B = lat_avg(B, latmin=latMin, latmax=latMax)

spd = 1
res1 = 'C128'
path1 = '/data/mgehne/FV3/replay_exps/C128/ERAI_free-forecast_C128/STREAM_2015103100/MODEL_DATA/SST_INITANOMALY2CLIMO-90DY/ALLDAYS/'
filebase1 = 'prcp_avg6h_fhr'  #720_C128_180x360.nc

fchrs = np.arange(0, 744, 24)
nfchr = len(fchrs)
exps = [0, 1]
explabels = ['trmm', res1]
nexps = len(exps)

PC = xr.DataArray(0., coords=[fchrs, exps], dims=['fchrs', 'exps'])

fi = 0
for ff in fchrs:
    fstr = f"{ff:02d}"
    print('Reading fhr='+fstr)
    ds = xr.open_dataset(path1 + filebase1 + fstr + '_C128_180x360.nc')
    data1 = ds.prcp
    data1 = data1.sel(time=slice(datestrt, datelast))
    data1 = data1*3600
    data1.attrs['units'] = 'mm/d'
    ds.close()
    data1 = lat_avg(data1, latmin=latMin, latmax=latMax)

    if fi == 0:
        ana1 = data1

    print('computing pattern correlation')
    corr = pattern_corr(ana1, data1)
    PC[fi,0] = corr
    corr = pattern_corr(B, data1)
    PC[fi,1] = corr
    fi += 1

plot_pattcorr(PC, explabels, plotpath, lats=latMin, latn=latMax)
