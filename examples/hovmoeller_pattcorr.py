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

res1 = 'c96'
path1 = '/data/mgehne/FV3/GAEA/control/'
filebase1 = 'prate_ave_f'
res2 = 'c384'
path2 = '/data/mgehne/FV3/GAEA/control_'+res2+'/'
filebase2 = 'prate_ave_regrid_T766_to_T126_f'

fchrs = np.arange(0, 121, 12)
nfchr = len(fchrs)
exps = ['1', '2', '12']
explabels = [res1, res2, res1+' vs '+res2]
nexps = 2

PC = xr.DataArray(0., coords=[fchrs, exps], dims=['fchrs', 'exps'])

fi = 0
for ff in fchrs:
    fstr = f"{ff:03d}"
    print('Reading fhr='+fstr)
    ds = xr.open_dataset(path1 + filebase1 + fstr + '.nc')
    data1 = ds.prate_ave
    data1 = lat_avg(data1, latmin=latMin, latmax=latMax)
    ds.close()
    ds = xr.open_dataset(path2 + filebase2 + fstr + '.nc')
    data2 = ds.prate_ave
    data2 = lat_avg(data2, latmin=latMin, latmax=latMax)
    ds.close()

    if fi == 0:
        ana1 = data1
        ana2 = data2

    print('computing pattern correlation')
    corr = pattern_corr(data1, data2)
    PC[fi,2] = corr
    corr = pattern_corr(ana1, data1)
    PC[fi,0] = corr
    corr = pattern_corr(ana2, data2)
    PC[fi,1] = corr
    fi += 1

plot_pattcorr(PC, explabels, plotpath, lats=latMin, latn=latMax)
