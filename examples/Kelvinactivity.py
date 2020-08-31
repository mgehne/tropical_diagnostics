import numpy as np
import xarray as xr
import sys
sys.path.append('../../')
"""
local scripts, if loading from a different directory include that with a '.' between
directory name and script name
"""
from diagnostics import CCEWactivity

"""
paths to plot and data directories
"""
plotpath = '../plots/'
eofpath = '../data/EOF/'

"""
Parameters to set for plotting Kelvin activity index.
"""
wave = 'MRG'
datestrt = '2015-12-01 00:00:00'
datelast = '2016-03-31 13:00:00'


print("reading ERAI data from file:")
spd = 1
ds = xr.open_dataset('/data/mgehne/ERAI/MetricsObs/precip.erai.sfc.1p0.'+str(spd)+'x.1979-2016.nc')
A = ds.precip
print("extracting time period:")
A = A.sel(time=slice(datestrt, datelast))
A = A.squeeze()
timeA = ds.time.sel(time=slice(datestrt, datelast))
ds.close()
A = A * 1000/4
A.attrs['units'] = 'mm/d'

print("project data onto wave EOFs")
waveactA = CCEWactivity.waveact(A, wave, eofpath, spd, '1p0', 181, 'annual')
print(waveactA.min(), waveactA.max())


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

print("project data onto wave EOFs")
waveactB = CCEWactivity.waveact(B, wave, eofpath, spd, '1p0', 181, 'annual')
print(waveactB.min(), waveactB.max())

print('reading model forecast from file:')
spd = 1
res1 = 'C128'
path1 = '/data/mgehne/FV3/replay_exps/C128/ERAI_free-forecast_C128/STREAM_2015103100/MODEL_DATA/SST_INITANOMALY2CLIMO-90DY/ALLDAYS/'
filebase1 = 'prcp_avg6h_fhr'  #720_C128_180x360.nc

fchrs = np.arange(0, 744, 24)
nfchr = len(fchrs)
exps = [0, 1, 2]
explabels = ['trmm', 'erai', res1]
nexps = len(exps)

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

    print('computing activity')
    wact1 = CCEWactivity.waveact(data1, wave, eofpath, spd, '1p0', 180, 'annual')

    if fi == 0:
        act = xr.DataArray(0., coords=[fchrs, exps, wact1['time']], dims=['fchrs', 'exps', 'time'])

    act[fi, 2, :] = wact1.values
    #act[fi, 3, :] = wact2.values
    act[fi, 0, :] = waveactB.values
    act[fi, 1, :] = waveactA.values

    print("plot activity")
    CCEWactivity.plot_activity(act[fi, :, :], wave, explabels, plotpath, ff)

    fi += 1

print("computing skill")
skill = CCEWactivity.wave_skill(act)
CCEWactivity.plot_skill(skill, wave, explabels[1::], plotpath)
