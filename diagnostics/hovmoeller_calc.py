import numpy as np
import xarray as xr
from scipy import signal

"""
Routines used to compute Hovmoeller diagrams and pattern correlation.

Included:

lat_avg:

pattern_corr:
"""


def lat_avg(data, latmin, latmax):
    """
    Compute latitudinal average for hovmoeller diagram.
    :param data: input data (time, lat, lon)
    :type data: xarray.Dataarray
    :param latmin: southern latitude for averaging
    :type latmin: float
    :param latmax: northern latitude for averaging
    :type latmax: float
    :return: data (time, lon)
    :rtype: xarray.Dataarray
    """
    data = data.sel(lat=slice(latmin, latmax))
    units = data.attrs['units']
    data = data.mean(dim='lat')
    data.attrs['units'] = units

    return data


def pattern_corr(a, b):
    """
    Compute the pattern correlation between two 2D (time, lon) fields
    :param a: (time, lon) data array
    :type a: float
    :param b: (time, lon) data array
    :type b: float
    :return: correlation
    :rtype: float
    """

    corr = signal.correlate2d(a, b, mode='full')

    return corr
