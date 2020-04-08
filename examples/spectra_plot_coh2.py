import numpy as np
from netCDF4 import Dataset
import xarray as xr
import sys

sys.path.append('../../')
from tropical_diagnostics.diagnostics import spacetime_plot as stp


pathdata = '../data/'
plotpath = '../plots/'

# plot layout parameters
flim = 0.5  # maximum frequency in cpd for plotting
nWavePlt = 20  # maximum wavenumber for plotting
contourmin = 0.05  # contour minimum
contourmax = 0.55  # contour maximum
contourspace = 0.05  # contour spacing
N = [1, 2]  # wave modes for plotting
source = ""

symmetry = "symm"      #("symm", "asymm", "latband")
nplot = 6
filenames = ['ERAI_TRMM_P_symm_1spd', 'FV3_TRMM_P_symm_1spd_fhr00', 'FV3_TRMM_P_symm_1spd_fhr24',
             'ERAI_P_D850_symm_1spd', 'FV3_P_D850_symm_1spd_fhr00', 'FV3_P_D850_symm_1spd_fhr24']
vars1 = ['ERAI P', 'FV3 P', 'FV3 P', 'ERAI P', 'FV3 P FH00', 'FV3 P FH24']
vars2 = ['TRMM', 'TRMM', 'TRMM', 'ERAI D850', 'FV3 D850 FH00', 'FV3 D850 FH24']
pp = 0

while pp < nplot:

    # read data from file
    var1 = vars1[pp]
    var2 = vars2[pp]
    fin = xr.open_dataset(pathdata + 'SpaceTimeSpectra_' + filenames[pp] + '.nc')
    STC = fin['STC'][:, :, :]
    wnum = fin['wnum']
    freq = fin['freq']

    #ifreq = np.where((freq[:] >= 0) & (freq[:] <= flim))
    #iwave = np.where(abs(wnum[:]) <= nWavePlt)

    STC[:, freq[:] == 0, :] = 0.
    STC = STC.sel(wnum=slice(-nWavePlt, nWavePlt))
    STC = STC.sel(freq=slice(0, flim))
    #STC = STC[:, :, iwave]
    #STC = STC[:, ifreq, :]
    coh2 = np.squeeze(STC[4, :, :])
    phs1 = np.squeeze(STC[6, :, :])
    phs2 = np.squeeze(STC[7, :, :])
    pow1 = np.squeeze(STC[0, :, :])
    pow2 = np.squeeze(STC[1, :, :])
    pow1.where(pow1 <= 0, drop=True)
    pow2.where(pow2 <= 0, drop=True)
    #pow1[pow1 <= 0] = np.nan
    #pow2[pow2 <= 0] = np.nan

    if pp == 0:
        Coh2 = np.empty([nplot, len(freq[ifreq]), len(wnum[iwave])])
        Phs1 = np.empty([nplot, len(freq[ifreq]), len(wnum[iwave])])
        Phs2 = np.empty([nplot, len(freq[ifreq]), len(wnum[iwave])])
        Pow1 = np.empty([nplot, len(freq[ifreq]), len(wnum[iwave])])
        Pow2 = np.empty([nplot, len(freq[ifreq]), len(wnum[iwave])])
        k = wnum[iwave]
        w = freq[ifreq]

    Coh2[pp, :, :] = coh2
    Phs1[pp, :, :] = phs1
    Phs2[pp, :, :] = phs2
    Pow1[pp, :, :] = np.log10(pow1)
    Pow2[pp, :, :] = np.log10(pow2)

    phstmp = Phs1
    phstmp = np.square(Phs1) + np.square(Phs2)
    phstmp = np.where(phstmp == 0, np.nan, phstmp)
    scl_one = np.sqrt(1 / phstmp)
    Phs1 = scl_one * Phs1
    Phs2 = scl_one * Phs2

    pp += 1

# plot coherence
stp.plot_coherence(Coh2, Phs1, Phs2, symmetry, source, var1, var2, plotpath, flim, 20, contourmin, contourmax,
                   contourspace, nplot, N)

exit()
