"""
Collection of functions and routines to compute vertical coherence profiles.
"""
import numpy as np
import xarray as xr
from tropical_diagnostics.diagnostics import spacetime as st

def vertical_coherence_comp(data1, data2, levels, nDayWin, nDaySkip, spd):
    """
     Main driver to compute vertical coherence profile. This can be called from
     a script that reads in filtered data and level data.
    :param data1: single level filtered precipitation input data
    :param data2: multi-level dynamical variable, dimension 1 needs to match the levels
    given in levels
    :param levels: vertical levels to compute coherence at
    :param nDayWin: number of time steps per window
    :param nDaySkip: number of time steps to overlap per window
    :param spd: number of time steps per day
    :return: CohAvg, CohMask, CohMat: Vertical profile, masked cross-spectra at all levels,
    full cross-spectra at all levels
    """

    # compute coherence - loop through levels

    # compute significant value of coherence based on distribution

    # average coherence across significant values

    # recompute phase angles of averaged cross-spectra

    # return output

