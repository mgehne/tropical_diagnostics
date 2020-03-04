import numpy as np
import xarray as xr
import sys
sys.path.append('../../')
"""
local scripts, if loading from a different directory include that with a '.' between
directory name and script name
"""
from tropical_diagnostics.diagnostics import CCEWactivity

"""
paths to plot and data directories
"""
plotpath = '../plots/'
eofpath = '../data/EOF/'

"""
Parameters to set for plotting Kelvin activity index.
"""
wave = 'Kelvin'
datestrt = '2016-01-06 00:00:00'
datelast = '2016-03-29 13:00:00'


print("reading ERAI data from file:")
spd = 2
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
waveactA = CCEWactivity.waveact(A, wave, eofpath, spd)
print(waveactA.min(), waveactA.max())


print("reading observed precipitation data from file:")
spd = 2
ds = xr.open_dataset('/data/mgehne/Precip/MetricsObs/precip.trmm.2x.1p0.v7a.fillmiss.comp.1998-2016.nc')
B = ds.precip
print("extracting time period:")
B = B.sel(time=slice(datestrt, datelast))
B = B.squeeze()
timeB = ds.time.sel(time=slice(datestrt, datelast))
ds.close()
B.attrs['units'] = 'mm/d'

print("project data onto wave EOFs")
waveactB = CCEWactivity.waveact(B, wave, eofpath, spd)
print(waveactB.min(), waveactB.max())

print('reading model forecast from file:')
spd = 2
res1 = 'c96'
path1 = '/data/mgehne/FV3/GAEA/control/'
filebase1 = 'prate_ave_1p0_f'
res2 = 'c384'
path2 = '/data/mgehne/FV3/GAEA/control_'+res2+'/'
filebase2 = 'prate_ave_1p0_f'

fchrs = np.arange(0, 121, 12)
nfchr = len(fchrs)
exps = [0, 1, 2, 3]
explabels = ['trmm', 'erai', res1, res2]

fi = 0
for ff in fchrs:
    fstr = f"{ff:03d}"
    print('Reading fhr='+fstr)
    ds = xr.open_dataset(path1 + filebase1 + fstr + '.nc')
    data1 = ds.prate_ave
    data1 = data1.sel(time=slice(datestrt, datelast))
    ds.close()
    ds = xr.open_dataset(path2 + filebase2 + fstr + '.nc')
    data2 = ds.prate_ave
    data2 = data2.sel(time=slice(datestrt, datelast))
    ds.close()

    print('computing activity')
    wact1 = CCEWactivity.waveact(data1, wave, eofpath, spd)
    wact2 = CCEWactivity.waveact(data2, wave, eofpath, spd)

    if fi == 0:
        act = xr.DataArray(0., coords=[fchrs, exps, wact1['time']], dims=['fchrs', 'exps', 'time'])

    act[fi, 2, :] = wact1.values
    act[fi, 3, :] = wact2.values
    act[fi, 0, :] = waveactB.values
    act[fi, 1, :] = waveactA.values

    print("plot activity")
    CCEWactivity.plot_activity(act[fi, :, :], wave, explabels, plotpath, ff)

    fi += 1

print("computing skill")
skill = CCEWactivity.wave_skill(act)
CCEWactivity.plot_skill(skill, wave, explabels[1::], plotpath)
