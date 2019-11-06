"""
Contains functions to project data onto CCEW EOFs.

List of functions:

proj:

comp_anom:

"""

import numpy as np
import xarray as xr

def proj(A, wave, eofpath, opt=False):
    # read EOFs from file
    if (wave == 'Kelvin' or wave == 'kelvin'):
        eofname = 'EOF_1-4_130-270E_-15S-15N_persiann_cdr_1p0_fillmiss8314_1983-2016_Kelvinband_'
    elif (wave == 'ER' or wave == 'er'):
        eofname = 'EOF_1-4_60-220E_-21S-21N_persiann_cdr_1p0_fillmiss8314_1983-2016_ERband_'

    ds = xr.open_dataset(eofpath + eofname + '01.nc')
    eoflat = ds.lat
    eoflon = ds.lon
    nlat = len(eoflat)
    nlon = len(eoflon)
    eofnum = np.arange(4)+1
    neof = 4
    month = np.arange(12)+1
    nmon = 12

    EOF = xr.DataArray('empty', coords=[month, eofnum, eoflat, eoflon], dims=['month', 'eofnum', 'lat', 'lon'])

    for ss in months:
        monthnum = f"{ss:02d}"
        ds = xr.open_dataset(eofpath+eofname+monthnum+'.nc')
        EOF[ss, :, :, :] = ds.eof

    # remove mean annual cycle

    # compute projection and activity index