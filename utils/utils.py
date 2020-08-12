"""
This is a collection of utility functions.

List of functions:

lonFlip:
    Flip longitudes from -180:180 to 0:360 or vice versa.

"""

import numpy as np
import xarray as xr

def lonFlip(data,lon):
    """
    Change the longitude coordinates from -180:180 to 0:360 or vice versa.
    :param data: Input xarray data array (time x lat x lon).
    :param lon: Longitude array of the input data.
    :return: dataflip
    """

    lonnew = lon.values

    if lonnew.min() < 0:
        # change longitude to 0:360
        ilonneg = np.where(lon<0)
        nlonneg = len(ilonneg[0])
        ilonpos = np.where(lon>=0)
        nlonpos = len(ilonpos[0])

        lonnew[0:nlonpos] = lon[ilonpos[0]].values
        lonnew[nlonpos:] = lon[ilonneg[0]].values + 360

        dataflip = xr.DataArray(np.roll(data, nlonneg, axis=2), dims=data.dims,
                          coords={'time': data['time'], 'lat': data['lat'], 'lon': lonnew})

    else:
        # change longitude to -180:180
        ilonneg = np.where(lon >= 180)
        nlonneg = len(ilonneg[0])
        ilonpos = np.where(lon < 180)
        nlonpos = len(ilonpos[0])

        lonnew[0:nlonneg] = lon[ilonneg[0]].values - 360
        lonnew[nlonneg:] = lon[ilonpos[0]].values

        dataflip = xr.DataArray(np.roll(data, nlonpos, axis=2), dims=data.dims,
                          coords={'time': data['time'], 'lat': data['lat'], 'lon': lonnew})

    return dataflip
