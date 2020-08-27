"""
Collection of functions and routines to compute vertical coherence profiles.

Content:

vertical_coherence_comp: driver script for vertical coherence calculations

coher_sig_dist: compute significant value of coherence from coherence distribution

cross_phase_2d: Compute phase angled from averaged cross-spectral components

"""
import numpy as np
import xarray as xr
from tropical_diagnostics.diagnostics import spacetime as st

def vertical_coherence_comp(data1, data2, levels, nDayWin, nDaySkip, spd, siglevel):
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
    :param siglevel: significance level
    :return: CohAvg, CohMask, CohMat: Vertical profile, masked cross-spectra at all levels,
    full cross-spectra at all levels
    """
    symmetries = ['symm', 'asymm']
    # compute coherence - loop through levels
    for ll in np.arange(0, len(levels), 1):
        print('processing level = '+str(levels[ll]))
        for symm in symmetries:
            y = st.get_symmasymm(data2[:, ll, :, :], data2['lat'], symm)
            x = st.get_symmasymm(data1, data1['lat'], symm)
            # compute coherence
            result = st.mjo_cross(x, y, nDayWin, nDaySkip)
            tmp = result['STC']  # , freq, wave, number_of_segments, dof, prob, prob_coh2
            try:
                CrossMat
            except NameError:
                # initialize cross-spectral array
                freq = result['freq']
                freq = freq * spd
                wnum = result['wave']
                dims = tmp.shape
                CrossMat = xr.DataArray(np.empty([len(levels), dims[0]*2, dims[1], dims[2]]),
                                  dims=['level', 'cross', 'freq', 'wave'],
                                  coords={'level': levels, 'cross': np.arange(0, 16, 1), 'freq': freq, 'wave': wnum})

            # write cross-spectral components to array
            if symm == 'symm':
                CrossMat[ll, 0::2, :, :] = tmp
            elif symm == 'asymm':
                CrossMat[ll, 1::2, :, :] = tmp

    # compute significant value of coherence based on distribution
    sigval = coher_sig_dist(CrossMat[:, 8:10, :, :].values, siglevel)
    print(str(siglevel*100)+"% significance coherence value: "+str(sigval))

    # mask cross-spectra where coherence < siglevel
    MaskArray = CrossMat[:, 8:9, :, :]
    MaskArray = np.where(MaskArray <= sigval, np.nan, 1)
    MaskAll = np.empty(CrossMat.shape)
    for i in np.arange(0,8,1):
        MaskAll[:, i*2:i*2+1, :, :] = MaskArray

    CrossMask = CrossMat * MaskAll

    # average coherence across significant values
    CrossAvg = np.nanmean(CrossMask.sel(freq=slice(0, 1), wave=slice(-20, 20)), axis=(2, 3))
    # recompute phase angles of averaged cross-spectra
    CrossAvg = cross_phase_2d(CrossAvg)
    CrossAvg = xr.DataArray(CrossAvg, dims=['level', 'cross'], coords={'level': levels, 'cross': np.arange(0, 16, 1)})

    # return output
    return CrossAvg, CrossMask, CrossMat


def coher_sig_dist(Coher, siglevel):
    """
    Compute the significant coherence level based on the distribution of coherence.
    Sorts the coherence values by size and picks the value corresponding to the siglevel
    percentile. E.g. for a siglevel or 0.95 it picks the value of coherence larger than
    95% of all the input values.
    :param Coher: numpy array containing all coherence values
    :return: sigval
    """
    # make a 1d array
    coher = Coher.flatten()
    # find all valid values
    coher = coher[~np.isnan(coher)]
    coher = coher[(0 <= coher) & (coher <= 1)]

    # sort array
    coher = np.sort(coher)
    nvals = len(coher)
    # find index of significant level
    isig = int(np.floor(nvals*siglevel))
    # read significant value
    sigval = coher[isig]

    return sigval

def cross_phase_2d(Cross):
    """
    Compute phase angles from cross spectra.
    :param Cross: 2d array with the cross-spectra components in dim=1
    :return: Cross with replaced phase angles
    """

    # read co- and quadrature-spectral components
    cxys = Cross[:, 4]
    cxya = Cross[:, 5]
    qxys = Cross[:, 6]
    qxya = Cross[:, 7]

    # compute phase angles
    pha_s = np.arctan2(qxys, cxys)
    pha_a = np.arctan2(qxya, cxya)

    # compute phase vectors
    v1s = -qxys / np.sqrt(np.square(qxys) + np.square(cxys))
    v2s = cxys / np.sqrt(np.square(qxys) + np.square(cxys))
    v1a = -qxya / np.sqrt(np.square(qxya) + np.square(cxya))
    v2a = cxya / np.sqrt(np.square(qxya) + np.square(cxya))

    # write phase angles and vectors to array
    Cross[:, 10] = pha_s
    Cross[:, 11] = pha_a
    Cross[:, 12] = v1s
    Cross[:, 13] = v1a
    Cross[:, 14] = v2s
    Cross[:, 15] = v2a

    return Cross
