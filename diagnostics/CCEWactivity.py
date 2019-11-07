"""
Contains functions to project data onto CCEW EOFs.

List of functions:

waveact:

rem_seas_cyc:

waveproj:

"""

import numpy as np
import xarray as xr


def waveact(data: object, wave: str, eofpath: str, opt=False):
    """
    Main script to compute the wave activity index.
    :param data: DataArray containing the raw data 
    :param wave: string describing the wave name
    :param eofpath: file path to EOFs
    :param opt: optional parameter, currently not used
    :return: DataArray containing the time series of wave activity
    """
    # read EOFs from file
    if (wave == 'Kelvin' or wave == 'kelvin'):
        eofname = 'EOF_1-4_130-270E_-15S-15N_persiann_cdr_1p5_fillmiss8314_1983-2016_Kelvinband_'
    elif (wave == 'ER' or wave == 'er'):
        eofname = 'EOF_1-4_60-220E_-21S-21N_persiann_cdr_1p5_fillmiss8314_1983-2016_ERband_'

    ds = xr.open_dataset(eofpath + eofname + '01.nc')
    nlat = len(ds.lat)
    nlon = len(ds.lon)
    eofnum = np.arange(4) + 1
    neof = 4
    month = np.arange(12) + 1
    nmon = 12
    ntim = len(data['time'])

    eofseas = xr.DataArray(0., coords=[month, eofnum, ds.lat, ds.lon], dims=['month', 'eofnum', 'lat', 'lon'])

    for ss in month:
        monthnum = f"{ss:02d}"
        ds = xr.open_dataset(eofpath + eofname + monthnum + '.nc')
        eofseas[ss - 1, :, :, :] = ds.eof
    ds.close()

    # remove mean annual cycle
    data_anom = rem_seas_cyc(data)

    # compute projection
    tswave = waveproj(data_anom, eofseas)

    # compute activity
    waveact = xr.DataArray(0., coords=[data_anom.time], dims=['time'])
    waveact.values = np.sum(np.square(tswave), 0)

    del data, data_anom

    return waveact


def rem_seas_cyc(data: object, opt: object = False) -> object:
    """
    Read in a xarray data array with time coordinate containing daily data. Compute anomalies from daily climatology.
    :type data: xr.DataArray
    :param data: xarray
    :param opt: optional parameter, not currently used
    :return: xr.DataArray containing anomalies from daily climatology
    """
    da = xr.DataArray(np.arange(len(data['time'])), coords=[data['time']], dims=['time'])
    month_day_str = xr.DataArray(da.indexes['time'].strftime('%m-%d'), coords=da.coords, name='month_day_str')
    time = data['time']

    data = data.rename({'time': 'month_day_str'})
    month_day_str = month_day_str.rename({'time': 'month_day_str'})
    data = data.assign_coords(month_day_str=month_day_str)
    clim = data.groupby('month_day_str').mean('month_day_str')

    data_anom = data.groupby('month_day_str') - clim
    data_anom = data_anom.rename({'month_day_str': 'time'})
    data_anom = data_anom.assign_coords(time=time)

    return data_anom


def waveproj(data_anom: object, eofseas: object):
    """
    Compute the projection onto the CCEW EOFS
    :param data_anom: anomalies of precipitation
    :param eofseas: xarray dataarray containing the monthly EOF patterns
    :return: wave time series projected onto each EOF
    """
    data_anom = data_anom.sel(lat=slice(eofseas.lat.min(), eofseas.lat.max()),
                              lon=slice(eofseas.lon.min(), eofseas.lon.max()))
    mm = data_anom.time.dt.month
    ntim = len(data_anom.time)
    neof = len(eofseas.eofnum)
    tswave: object = xr.DataArray(0., coords=[eofseas.eofnum, data_anom.time], dims=['eofnum', 'time'])
    for tt in range(ntim):
        eof = eofseas[mm[tt] - 1, :, :, :]
        for ee in range(neof):
            tswave[ee, tt] = eof[ee, :, :] @ data_anom[tt, :, :]

    return tswave
