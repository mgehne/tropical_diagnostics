"""
Example script to plot vertical coherence profiles for several variables. The user needs to first
generate the vertical coherence data by running vertical_coherence.py. This script reads in data for
all variables specified below and plots the average vertical coherence profile and phase angle.
"""
import xarray as xr
import numpy as np
import tropical_diagnostics.vertical_coherence_plotly as vcp


var1 = "precip"  # variable name of data in the precipitation file
vars2 = ["uwnd", "air", "div", "shum", "vwnd"]  # variable names of data in the second file
labels = vars2

source1 = "TRMM"  # ERAI, ERA5, TRMM
source2 = "ERA5"

wave1 = "MJO"  # "kelvin", "MRG", "ER", "MJO", "IG0"

RES = "2p5"  # spatial resolution of the data
spd = 1  # data is spd x daily

# first and last date format: yyyymmddhh
datemin = '2007-01-01'
datemax = '2010-12-31'
yearmin = datemin[0:4]
yearmax = datemax[0:4]

# significance level for the coherence plots
sigstr = 99.
siglev = sigstr / 100

# latBound
latN = 20
latS = -latN

# path to read data from - this needs to be computed using vertical_coherence.py
# prior to running this script
pathout = '/data/mgehne/VerticalCoherence/'
plotpath = "/Users/mgehne/Projects/Diagnostics/Plots/VerticalCoherence/"

# setting up filenames
outfilepre = "CoherenceVertical_python_" + RES + "_" + str(spd) + "x_" + source1 + var1 + wave1 + "_"
outfilepost = "_" + str(datemin) + "-" + str(datemax) + "_" + str(latS) + "-" + str(latN) + "_sigMask"
outfileSpectrapre = "CoherenceVertical_SpaceTime_python_" + RES + "_" + str(spd) + "x_" + source1 + var1 + wave1 + "_"

# loop through variables and read average coherence
for vv in np.arange(0, len(vars2), 1):
    var2 = vars2[vv]
    print('==========================')
    print(var2)
    if wave1 == "kelvin":
        Symmetry = "symm"
        if var2 == "vwnd":
            Symmetry = "anti-symm"
    if wave1 == "ER":
        Symmetry = "symm"
        if var2 == "vwnd":
            Symmetry = "anti-symm"
    if wave1 == "MJO":
        Symmetry = "symm"
        if var2 == "vwnd":
            Symmetry = "anti-symm"
    if wave1 == "MRG":
        Symmetry = "anti-symm"
        if var2 == "vwnd":
            Symmetry = "symm"
    if wave1 == "":
        Symmetry = "anti-symm"

    labels[vv] = labels[vv] + "  " + Symmetry

    if Symmetry == "symm":
        n = 8  # symmetric coherence spectra
        m = 12  # symmetric phase 1 angle
        k = 14  # symmetric phase 2 angle
    else:
        n = 9  # anti-symmetric coherence spectra
        m = 13  # anti-symmetric phase 1 angle
        k = 15  # anti-symmetric phase 2 angle

    outfile = outfilepre + source2 + var2 + outfilepost + ".nc"
    ds = xr.open_dataset(pathout + outfile)
    CrossAvg = ds['CrossAvg']
    CohAvg = CrossAvg[:, n]
    PxAvg = CrossAvg[:, m]
    PyAvg = CrossAvg[:, k]
    try:
        CohAll
    except NameError:
        levels = ds['level']
        CohAll = xr.DataArray(np.empty([len(vars2), len(levels)]),
                              dims=['vars', 'level'], coords={'vars': vars2, 'level': levels})
        PxAll = xr.DataArray(np.empty([len(vars2), len(levels)]),
                             dims=['vars', 'level'], coords={'vars': vars2, 'level': levels})
        PyAll = xr.DataArray(np.empty([len(vars2), len(levels)]),
                             dims=['vars', 'level'], coords={'vars': vars2, 'level': levels})
    CohAll[vv, :] = CohAvg
    PxAll[vv, :] = PxAvg
    PyAll[vv, :] = PyAvg

plotname = "SpTiCoherenceVertical_pythontest_" + RES + "_" + str(
    spd) + "x_" + wave1 + "_" + source1 + "_" + source2 + "_" + str(latS) + "to" + str(latN) + "_" + str(
    yearmin) + "-" + str(yearmax) + "_sigMask"
vcp.plot_vertcoh(CohAll, PxAll, PyAll, levels, labels, wave1 + var1, plotname, plotpath, latS, latN)




