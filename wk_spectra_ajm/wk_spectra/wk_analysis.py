# -*- coding: utf-8 -*-

"""
Utilities for plotting the Wheeler-Kiladis Space-Time Spectra
Created by: Alejandro Jaramillo
ajaramillomoreno@gmail.com
2018

This code corresponds to a revision of a previous python code created by me at
my Internship between September 2014 - August 2015 at NOAA Boulder, Colorado
with George Kiladis.

I based the original code on the NCL scripts created by Dennis Shea at NCAR.
The new version strongly departs from the NCL version using now more python
libraries and clearing the code.  In particular, the tapering in the time
windows is made different so slighly differences in the results are expected
when comparing with the same analysis made using NCL .

References:

Wheeler, M., & Kiladis, G. N. (1999).
Convectively Coupled Equatorial Waves: Analysis of Clouds and Temperature in
the Wavenumber–Frequency Domain. Journal of the Atmospheric Sciences,
56(3), 374–399.
https://doi.org/10.1175/1520-0469(1999)056<0374:CCEWAO>2.0.CO;2

Kiladis, G. N., Wheeler, M. C., Haertel, P. T., Straub, K. H.,
& Roundy, P. E. (2009). Convectively coupled equatorial waves.
Reviews of Geophysics, 47(2), RG2003.
https://doi.org/10.1029/2008RG000266

Wheeler, M. C., & Nguyen, H. (2015).
TROPICAL METEOROLOGY AND CLIMATE | Equatorial Waves.
In Encyclopedia of Atmospheric Sciences (pp. 102–112). Elsevier.
https://doi.org/10.1016/B978-0-12-382225-3.00414-X

Hayashi, Y., (1971).
A Generalized Method of Resolving Disturbances into
Progressive and Retrogressive Waves by Space and
Fourier and TimeCross Spectral Analysis.
J. Meteor. Soc. Japan, 1971, 49: 125-128.
;-----------------------------------------------
"""
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import netCDF4 as nc
import wk_spectra.matsuno_plot as mp
import os
from wk_spectra.errors import InputError

import seaborn as sns
sns.set_style("whitegrid")

def sym_asym(array_in):
    """
    Function that decompose an array (supports only array_in(lat,lon)),
    into its symmetric and asymmetric parts about the equator.

    The asymmetric part is stored in the Northern Hemisphere as
    SymAsym[lat] = (array_in[lat]-array_in[-lat])/2,
    while the symmetric part is stored in the Southern Hemisphere as
    SymAsym[-lat] = (array_in[lat]+array_in[-lat])/2

    :param array_in:
        Input Aray
    :type array_in: Numpy Array
    :return: SymAsym
    :rtype: Numpy Array
    """
    (nlat,nlon)  = array_in.shape
    N = nlat//2

    #Copy array with same shape as array_in
    #values will be overwritten except when the eq is
    #present in the array
    SymAsym = np.copy(array_in)

    for i in range(N):
        # Save symmetric part in the Southern Hemisphere
        SymAsym[i,:] = 0.5*(array_in[nlat-1-i,:] + array_in[i,:])
        # Save antisymmetric part in the Northern Hemisphere
        SymAsym[nlat-1-i,:] = 0.5*(array_in[nlat-1-i,:] - array_in[i,:])
        # Notice that the equator if present is untouched

    return SymAsym

def decompose_symasym(array_in):
    """
    Function that decompose an array (supports array_in(lat,lon),
    array_in(time,lat,lon) and array_in(time,level,lat,lon)) into its symmetric
    and asymmetric parts about the equator.

    The asymmetric part is stored in the Northern Hemisphere as
    SymAsym[lat] = (array_in[lat]-array_in[-lat])/2,
    while the symmetric part is stored in the Southern Hemisphere as
    SymAsym[-lat] = (array_in[lat]+array_in[-lat])/2

    :param array_in:
        Input Aray
    :type array_in: Numpy Array
    :return: SymAsym
    :rtype: Numpy Array
    """
    #Check the dimensions of the array to find out if it has the structure of
    #array_in(lat,lon), array_in(time,lat,lon) or array_in(time,level,lat,lon)
    dim_array  = array_in.shape
    rank_array = len(dim_array)

    if (rank_array>=5):
        raise InputError("Decompose_SymAsym: currently supports up to 4D: rank = {}D"\
                        .format(rank_array))

    #Copy array with same shape as array_in
    #values will be overwritten except when the eq is
    #present in the array

    SymAsym = np.copy(array_in)

    if (rank_array==1):
        for i in range(N):
            # Save symmetric part in the Southern Hemisphere
            SymAsym[i] = 0.5*(array_in[nlat-1-i] + array_in[i])
            # Save antisymmetric part in the Northern Hemisphere
            SymAsym[nlat-1-i] = 0.5*(array_in[nlat-1-i] - array_in[i])
            # Notice that the equator if present is untouched

    if (rank_array==2):
        SymAsym = sym_asym(array_in)

    if (rank_array==3):
        ntim = dim_array[0]
        for t in range(ntim):
            SymAsym[t,:,:] = sym_asym(array_in[t,:,:])

    if (rank_array==4):
        ntim = dim_array[0]
        nlevel = dim_array[1]
        for t in range(ntim):
            for l in range(nlevel):
                SymAsym[t,l,:,:] = sym_asym(array_in[t,l,:,:])

    return SymAsym

def sampling_vars(array_in,spd,nDayWin,nDaySkip):

    # Supports only array_in(time,lat,lon)
    ntim,nlat,nlon = array_in.shape

    if ((ntim%spd)!=0):
        raise InputError("Input array must have complete days only ntim%spd = {}".format(ntim%spd))

    # Number of days
    nDayTot = ntim//spd
    # Number of samples per temporal window
    nSampWin = nDayWin*spd
    # Number of samples to skip between window segments.
    # Negative number means overlap
    nSampSkip = nDaySkip*spd

    return (ntim,nlat,nlon,nDayTot,nSampWin,nSampSkip)

def remove_dominant_signals(array_in,spd,nDayWin,nDaySkip):
    """
    This function removes the dominant signals by removing the long term
    linear trend (conserving the mean) and by eliminating the annual cycle
    by removing all time periods less than a corresponding critical
    frequency.
    """
    ntim,nlat,nlon,nDayTot,nSampWin,nSampSkip = sampling_vars(array_in,spd,nDayWin,nDaySkip)

    # Critical frequency
    fCrit   = 1./nDayWin

    # Remove long term linear trend
    long_mean = np.mean(array_in,axis=0)
    detrend = signal.detrend(array_in,axis=0,type='linear')
    #remove just trend conserving the mean
    array_dt = detrend+long_mean

    if (nDayTot>=365):
        fourier = np.fft.rfft(array_dt,axis=0)
        fourier_mean = np.copy(fourier[0,:,:])
        freq = np.fft.rfftfreq(ntim,1./spd)
        ind = np.where(freq<=fCrit)[0]
        fourier[ind,:,:] = 0.0
        fourier[0,:,:] = fourier_mean
        array_dt = np.fft.irfft(fourier,axis=0)

    return array_dt

def smooth121(array_in):
    """
    Smoothing function that takes a 1D array a pass it through a 1-2-1 filter.
    This function is a modified version of the wk_smooth121 from NCL.
    The weights for the first and last points are  3-1 (1st) or 1-3 (last) conserving the total sum.
    :param array_in:
        Input array
    :type array_in: Numpy array
    :return: array_out
    :rtype: Numpy array
    """

    temp = np.copy(array_in)
    array_out = np.copy(temp)*0.0
    weights = np.array([1.0,2.0,1.0])/4.0
    sma = np.convolve(temp, weights,'valid')
    array_out[1:-1] = sma

    # Now its time to correct the borders
    if (np.isnan(temp[1])):
        if (np.isnan(temp[0])):
            array_out[0] = np.nan
        else:
            array_out[0] = temp[0]
    else:
        if (np.isnan(temp[0])):
            array_out[0] = np.nan
        else:
            array_out[0] = (temp[1]+3.0*temp[0])/4.0
    if (np.isnan(temp[-2])):
        if (np.isnan(temp[-1])):
            array_out[-1] = np.nan
        else:
            array_out[-2] = array_out[-2]
    else:
        if (np.isnan(temp[-1])):
            array_out[-1] = np.nan
        else:
            array_out[-1] = (temp[-2]+3.0*temp[-1])/4.0

    return array_out

def spectral_coefficients(array_in,spd,nDayWin,nDaySkip):

    ntim,nlat,nlon,nDayTot,nSampWin,nSampSkip = sampling_vars(array_in,spd,nDayWin,nDaySkip)

    # Test if there is enought time data for the analysis
    if (ntim<nSampWin):
        raise InputError("The available number of days is less than the sample window")
    else:
        # Count the number of available samples
        nWindow = (ntim-nSampWin)//(nSampWin+nSampSkip)+1


    # Test if longitude and time dimensions are even.
    # The fft algorith gives the nyquist frequency one time for even and two times
    # for odd dimensions
    if (nSampWin%2==0): #if time is even
        if (nlon%2==0): # and longitude is even also
            peeAS = np.zeros((nWindow,nSampWin+1,nlat,nlon+1),dtype='c16')
        else: # but longitude is odd
            peeAS = np.zeros((nWindow,nSampWin+1,nlat,nlon),dtype='c16')
    else: #if time is odd
        if (nlon%2==0): # but longitude is even
            peeAS = np.zeros((nWindow,nSampWin,nlat,nlon+1),dtype='c16')
        else: # but longitude is odd also
            peeAS = np.zeros((nWindow,nSampWin,nlat,nlon),dtype='c16')

    # Create a tapering window(nSampWin,nlat,nlon) using the hanning function.
    # For more information about the hanning function see:
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.hanning.html
    # Note: Hanning function is different from Hamming function.
    tapering_window = np.repeat(np.hanning(nSampWin), nlat*nlon).reshape(nSampWin,nlat,nlon)

    ntStrt = 0
    ntLast = nSampWin

    for nw in range(nWindow):
        # Detrend temporal window
        temp_window = signal.detrend(array_in[ntStrt:ntLast,:,:],axis=0,type='linear')
        # Taper temporal window in the time dimension
        temp_window = temp_window*tapering_window
        # Apply fft to time and longitude
        fourier_fft = np.fft.fft2(temp_window,axes=(0,2))
        # normalize by # time samples
        fourier_fft = fourier_fft/(nlon*nSampWin)
        # fourier_fft(nSampWin,nlat,nlon) contains the
        # complex space-time spectrum for each latitude


        # Special reordering to resolve the Progressive and Retrogressive waves
        # based on Hayashi (1971).

        fourier_fft = np.fft.fftshift(fourier_fft, axes=(0,2))


        if (nSampWin%2==0): #if time is even
            if (nlon%2==0): # and longitude is even also
                varspacetime = np.zeros((nSampWin+1,nlat,nlon+1),dtype='c16')
                varspacetime[:nSampWin,:,:nlon] = fourier_fft
                varspacetime[nSampWin,:,:] = varspacetime[0,:,:]
                varspacetime[:,:,nlon] = varspacetime[:,:,0]
            else: # but longitude is odd
                varspacetime = np.zeros((nSampWin+1,nlat,nlon),dtype='c16')
                varspacetime[:nSampWin,:,:] = fourier_fft
                varspacetime[nSampWin,:,:] = varspacetime[0,:,:]
        else: #if time is odd
            if (nlon%2==0): # but longitude is even
                varspacetime = np.zeros((nSampWin,nlat,nlon+1),dtype='c16')
                varspacetime[:,:,:nlon] = fourier_fft
                varspacetime[:,:,nlon] = varspacetime[:,:,0]
            else: # but longitude is odd also
                varspacetime = np.zeros((nSampWin,nlat,nlon),dtype='c16')
                varspacetime[:,:,:] = fourier_fft

        fourier_fft = varspacetime

        # To correct that a positive freq in
        # fourier corresponds to a negative wave freq
        # i.e. Fourier -> e^i(kx+wt) != Wave -> e^i(kx-wt)
        fourier_fft = fourier_fft[:,:,::-1]
        # Save the Fourier Coefficients for each window
        peeAS[nw,:,:,:] = fourier_fft
        # Set index for next temporal window
        ntStrt = ntLast+nSampSkip
        ntLast = ntStrt+nSampWin
        del fourier_fft, temp_window

    wavefft = np.arange(-int(nlon/2),int(nlon/2)+1.,1.)
    freqfft = np.arange(-1.*int(nDayWin*spd/2),1.*int(nDayWin*spd/2)+1.,1)/(nDayWin)# Calculate the power spectrum
    return (wavefft,freqfft,peeAS)

def separate_power(power,nlat,nSampWin,wavefft,freqfft):

    # Separate the antisymmetric and symmetric power spectra
    if (nlat%2==0):
        # The array does not contain the Equator
        psumanti = np.sum(power[:,:,int(nlat/2):nlat,:],axis=2)
        psumsym  = np.sum(power[:,:,:int(nlat/2),:],axis=2)
    else:
        # The array contains the Equator, so it is skipped
        psumanti = np.sum(power[:,:,int(nlat/2)+1:nlat,:],axis=2)
        psumsym  = np.sum(power[:,:,:int(nlat/2),:],axis=2)

    # Now psumanti[window,freq,lon] and psumsym[window,freq,lon]
    # Since summing over half the array (symmetric,asymmetric) the
    # total variance is 2x the half sum
    psumanti = 2.0*psumanti
    psumsym  = 2.0*psumsym

    # Standarizing between the number of latitudes.
    # The Equator was skipped in the previous step so care
    # must be taken when the Equator is present.
    if (nlat%2==0):
        # The array does not contain the Equator
        psumanti = psumanti/nlat
        psumsym  = psumsym/nlat
    else:
        # The array contains the Equator, so the number of latitudes is
        # nlat-1 (the Equator is removed)
        psumanti = psumanti/(nlat-1)
        psumsym  = psumsym/(nlat-1)

    # Calculating the mean between time windows
    psumanti = np.mean(psumanti,axis=0)
    psumsym = np.mean(psumsym,axis=0)
    # Now psumanti[freq,lon] and psumsym[freq,lon]

    # Find where freq is equal to 0
    indfreq0 = np.where(freqfft==0.)[0]

    # Set the mean to missing to match original code
    psumanti[indfreq0,:] = np.nan
    psumsym[indfreq0,:] = np.nan

    # Apply smoothing to the spectrum. smooth over limited wave numbers
    # Smoothing in frequency only
    maxwav4smth = 27
    indwvsmth = np.where(np.logical_and(wavefft>=-maxwav4smth, wavefft<=maxwav4smth))[0]

    for wv in indwvsmth:
        psumanti[int(nSampWin/2):,wv] = smooth121(psumanti[int(nSampWin/2):,wv])
        psumsym[int(nSampWin/2):,wv] = smooth121(psumsym[int(nSampWin/2):,wv])

    return (psumanti,psumsym)

def derive_background(power,nlat,nSampWin,wavefft,freqfft):

    # Now derive the background spectrum (red noise)
    # Sum power over all latitude
    # Apply smoothing to the spectrum. This smoothing DOES include
    # wavenumber zero.

    if (nlat%2==0):
        psumb = np.sum(power,axis=2)  # sum over all latitudes
    else:
        tmp   = power
        tmp[:,:,int(nlat/2),:] = 0.0 # Remove contribution from equator
        psumb = np.sum(tmp,axis=2)
        del tmp

    # Standarizing between the number of latitudes, the equator is not used.
    if (nlat%2==0):
        psumb=psumb/nlat
    else:
        psumb=psumb/(nlat-1)

    # Calculating the mean between time windows
    psumb = np.mean(psumb,axis=0)

    # Find where freq is equal to 0
    indfreq0 = np.where(freqfft==0.)[0]
    # Set the mean to missing to match original code
    psumb[indfreq0,:] = np.nan

    # Apply smoothing to the spectrum over limited wave numbers
    maxwav4smth = 27
    indwvsmth = np.where(np.logical_and(wavefft>=-maxwav4smth, wavefft<=maxwav4smth))[0]


    for tt in np.arange(int(nSampWin/2)+1,nSampWin+1):
        if (freqfft[tt]<0.1):
            for i in range(5):
                psumb[tt,indwvsmth] = smooth121(psumb[tt,indwvsmth])
        if (np.logical_and(freqfft[tt]>=0.1,freqfft[tt]<0.2)):
            for i in range(10):
                psumb[tt,indwvsmth] = smooth121(psumb[tt,indwvsmth])
        if (np.logical_and(freqfft[tt]>=0.2,freqfft[tt]<0.3)):
            for i in range(20):
                psumb[tt,indwvsmth] = smooth121(psumb[tt,indwvsmth])
        if (freqfft[tt]>=0.3):
            for i in range(40):
                psumb[tt,indwvsmth] = smooth121(psumb[tt,indwvsmth])

    # smooth frequency up to .8 cycles per day
    indfreqsmth = np.where(np.logical_and(freqfft>0,freqfft<=0.8))[0]

    for wv in indwvsmth:
        for i in range(10):
            psumb[indfreqsmth,wv] = smooth121(psumb[indfreqsmth,wv])

    return psumb

def raw_plotter(X,Y,array_in,max_wn_plot,max_freq_plot,figsize,contour_range,cmap,text_size,freq_lines,cpd_lines):
    fig, ax = plt.subplots(figsize=figsize)
    cset  = ax.contourf(X,Y,array_in,contour_range,extend='both',cmap=cmap)
    cset1 = ax.contour(X,Y,array_in,contour_range,colors='k')
    plt.colorbar(cset,ax=ax)
    ax.xaxis.set_tick_params(labelsize=text_size)
    ax.yaxis.set_tick_params(labelsize=text_size)

    ax.axvline(x=0, color='k', linestyle='--')

    ax.set_xlabel('Zonal Wavenumber',size=text_size,fontweight='bold')
    ax.set_ylabel('Frequency (CPD)',size=text_size,fontweight='bold')
    ax.text(max_wn_plot-2*0.25*max_wn_plot,-0.01,'EASTWARD',fontweight='bold',fontsize=text_size-2)
    ax.text(-max_wn_plot+0.25*max_wn_plot,-0.01,'WESTWARD',fontweight='bold',fontsize=text_size-2)
    ax.set_xlim((-max_wn_plot,max_wn_plot))
    ax.set_ylim((0.02,max_freq_plot))

    if freq_lines:
        for d in cpd_lines:
            if ((1./d)<=max_freq_plot):
                ax.axhline(y=1./d,color='k', linestyle='--')
                ax.text(-max_wn_plot+0.2,(1./d+0.01),str(d)+' days',\
                        size=text_size,bbox={'facecolor':'white','alpha':0.9,\
                        'edgecolor':'none'})

    return fig,ax

def background_removed_plotter(X,Y,array_in,max_wn_plot,max_freq_plot,figsize,contour_range,\
                               contour_range_lines,cmap,text_size,freq_lines,cpd_lines,\
                               matsuno_lines,he,meridional_modes,is_sym,labels):

    fig, ax = plt.subplots(figsize=figsize)
    cset  = ax.contourf(X,Y,array_in,contour_range,extend='both',cmap=cmap)
    cset1 = ax.contour(X,Y,array_in,contour_range_lines,colors='k')
    plt.colorbar(cset,ax=ax)
    ax.xaxis.set_tick_params(labelsize=text_size)
    ax.yaxis.set_tick_params(labelsize=text_size)

    ax.axvline(x=0, color='k', linestyle='--')

    ax.set_xlabel('Zonal Wavenumber',size=text_size,fontweight='bold')
    ax.set_ylabel('Frequency (CPD)',size=text_size,fontweight='bold')
    ax.text(max_wn_plot-2*0.25*max_wn_plot,-0.01,'EASTWARD',fontweight='bold',fontsize=text_size-2)
    ax.text(-max_wn_plot+0.25*max_wn_plot,-0.01,'WESTWARD',fontweight='bold',fontsize=text_size-2)
    ax.set_xlim((-max_wn_plot,max_wn_plot))
    ax.set_ylim((0.02,max_freq_plot))

    if freq_lines:
        for d in cpd_lines:
            if ((1./d)<=max_freq_plot):
                ax.axhline(y=1./d,color='k', linestyle='--')
                ax.text(-max_wn_plot+0.2,(1./d+0.01),str(d)+' days',\
                        size=text_size,bbox={'facecolor':'white','alpha':0.9,\
                        'edgecolor':'none'})


    if matsuno_lines:
        matsuno_modes = mp.matsuno_modes_wk(he=he,n=meridional_modes,max_wn=max_wn_plot)

        for key in matsuno_modes:
            if is_sym:
                ax.plot(matsuno_modes[key]['Kelvin(he={}m)'.format(key)],color='k',linestyle='--')
                ax.plot(matsuno_modes[key]['ER(n=1,he={}m)'.format(key)],color='k',linestyle='--')
                ax.plot(matsuno_modes[key]['EIG(n=1,he={}m)'.format(key)],color='k',linestyle='--')
                ax.plot(matsuno_modes[key]['WIG(n=1,he={}m)'.format(key)],color='k',linestyle='--')
            else:
                ax.plot(matsuno_modes[key]['MRG(he={}m)'.format(key)],color='k',linestyle='--')
                ax.plot(matsuno_modes[key]['EIG(n=0,he={}m)'.format(key)],color='k',linestyle='--')


        if labels:

            key = list(matsuno_modes.keys())[len(list(matsuno_modes.keys()))//2]
            wn = matsuno_modes[key].index.values

            if is_sym:

                # Print Kelvin Label
                i = int((len(wn)/2)+0.3*(len(wn)/2))
                i, = np.where(wn == wn[i])[0]
                ax.text(wn[i]-1,matsuno_modes[key]['Kelvin(he={}m)'.format(key)].iloc[i],'Kelvin', \
                bbox={'facecolor':'white','alpha':0.9,'edgecolor':'none'},fontsize=text_size+4)

                # Print ER Label
                i = int(0.7*(len(wn)/2))
                i = np.where(wn == wn[i])[0]
                ax.text(wn[i]-1,matsuno_modes[key]['ER(n=1,he={}m)'.format(key)].iloc[i]+0.01,'ER', \
                bbox={'facecolor':'white','alpha':0.9,'edgecolor':'none'},fontsize=text_size+1)

                key2 = list(matsuno_modes.keys())[0]
                wn2 = matsuno_modes[key].index.values

                # Print EIG Label
                i = int((len(wn2)/2)+0.3*(len(wn2)/2))
                i, = np.where(wn2 == wn2[i])[0]
                ax.text(wn2[i]-1,matsuno_modes[key2]['EIG(n=1,he={}m)'.format(key2)].iloc[i],'EIG', \
                bbox={'facecolor':'white','alpha':0.9,'edgecolor':'none'},fontsize=text_size+1)

                # Print WIG Label
                i = int(0.55*(len(wn2)/2))
                i, = np.where(wn2 == wn2[i])[0]
                ax.text(wn2[i]-1,matsuno_modes[key2]['WIG(n=1,he={}m)'.format(key2)].iloc[i],'WIG', \
                bbox={'facecolor':'white','alpha':0.9,'edgecolor':'none'},fontsize=text_size+1)

            else:

                # Print EIG(n=0) Label
                i = int((len(wn)/2)+0.1*(len(wn)/2))
                i, = np.where(wn == wn[i])[0]
                ax.text(wn[i]-1,matsuno_modes[key]['EIG(n=0,he={}m)'.format(key)].iloc[i],'EIG(n=0)', \
                bbox={'facecolor':'white','alpha':0.9,'edgecolor':'none'},fontsize=text_size+1)

                # Print MRG Label
                i = int(0.7*(len(wn)/2))
                i, = np.where(wn == wn[i])[0]
                ax.text(wn[i]-1,matsuno_modes[key]['MRG(he={}m)'.format(key)].iloc[i],'MRG', \
                bbox={'facecolor':'white','alpha':0.9,'edgecolor':'none'},fontsize=text_size+1)


    return fig,ax

class wk_analysis(object):

    def __init__(self,name=None):
        self.data_status = "Empty"
        self.analysis_status = "Empty"
        self.plots_available = "No"
        if name is not None:
            self.name = str(name)
        print("wk_analysis object initiated and ready to receive data.")

    def print_status(self):
        no_status = ['data','wk_spectra','plots']
        for i in self.__dict__.keys():
            if i not in no_status:
                print(i + " = " + str(self.__dict__[i]))
        return None

    def import_netcdf(self,file,varname,latname='latitude',latBound=15):
        if (self.data_status=="Empty"):
            self.data_status = "NetCDF File"
            self.file = str(file)
            self.varname = varname
            self.latname = latname
            self.latBound = latBound
            f = nc.Dataset(file, 'r')
            array_out = f.variables[varname][:]
            latitudes = f.variables[latname][:]
            lat_ind = np.where(np.logical_and(latitudes>=-latBound,latitudes<=latBound))[0]
            array_out = array_out[:,lat_ind,:]
            self.data = array_out
            print('The file {} has been imported'.format(self.file))
        else:
            raise InputError("You have already loaded some data")

        return None

    def import_array(self,array_in,varname):
        if (self.data_status=="Empty"):
            self.data_status = "Numpy array"
            self.file = "array"
            self.varname = varname
            self.data = array_in
            print('An array has been imported')
        else:
            raise InputError("You have already loaded some data")
        return None

    def wheeler_kiladis_spectra(self,spd,nDayWin,nDaySkip,max_freq=0.5,max_wn=20):
        if (self.data_status=="Empty"):
            raise InputError("You need to input some data first.")
        else:
            if (self.analysis_status=="Empty"):
                array_in = self.data
                self.spd = spd
                self.nDayWin = nDayWin
                self.nDaySkip = nDaySkip
                self.max_freq = max_freq
                self.max_wn = max_wn

                ntim,nlat,nlon,nDayTot,nSampWin,nSampSkip = sampling_vars(array_in,spd,nDayWin,nDaySkip)

                # Remove dominant signals
                array_dt = remove_dominant_signals(array_in,spd,nDayWin,nDaySkip)

                # Decompose in Symmetric and Antisymmetric components
                array_as = decompose_symasym(array_dt)

                wavefft,freqfft,peeAS = spectral_coefficients(array_as,spd,nDayWin,nDaySkip)

                # Calculate the power spectrum
                power = (abs(peeAS))**2 # power[window,freq,lat,lon]

                psumanti,psumsym = separate_power(power,nlat,nSampWin,wavefft,freqfft)

                psumb = derive_background(power,nlat,nSampWin,wavefft,freqfft)

                # Cropping the output
                indwave = np.where(np.logical_and(wavefft>=-max_wn,wavefft<=max_wn))[0]
                indfreq = np.where(np.logical_and(freqfft>0,freqfft<=max_freq))[0]

                wavefft = wavefft[indwave]
                freqfft = freqfft[indfreq]

                psumanti = psumanti[indfreq,:]
                psumanti = psumanti[:,indwave]

                psumsym = psumsym[indfreq,:]
                psumsym = psumsym[:,indwave]

                psumb = psumb[indfreq,:]
                psumb = psumb[:,indwave]

                # Log10 scaling
                psumanti_log = np.log10(psumanti)
                psumsym_log = np.log10(psumsym)
                psumb_log = np.log10(psumb)

                psumanti_r = psumanti/psumb
                psumsym_r = psumsym/psumb

                self.wk_spectra = {
                            'psumanti':psumanti,
                            'psumsym':psumsym,
                            'psumb':psumb,
                            'psumanti_log':psumanti_log,
                            'psumsym_log':psumsym_log,
                            'psumb_log':psumb_log,
                            'psumanti_r':psumanti_r,
                            'psumsym_r':psumsym_r,
                            'wavefft':wavefft,
                            'freqfft':freqfft,
                            'spd':spd,
                            'nDayWin':nDayWin,
                            'nDaySkip':nDaySkip,
                            'max_freq':max_freq,
                            'max_wn':max_wn,
                            'varname':self.varname
                            }
                self.analysis_status = "Complete"
                print("The Wheeler-Kiladis Analysis is complete.")
            else:
                raise InputError("You have already done the analysis.")
            return None

    def plot_raw(self,figsize=(10,8),text_size=12,contour_range=(-1.9,-0.15,0.1),\
                cmap='Reds',max_freq_plot=None,max_wn_plot=None,freq_lines=True,\
                cpd_lines=[3,6,30]):

        if (self.plots_available == "No"):
            self.plots_available = "Yes"
            self.plots = {}

        if max_freq_plot is None:
            self.max_freq_plot = self.max_freq
        else:
            self.max_freq_plot = max_freq_plot
        if max_wn_plot is None:
            self.max_wn_plot = self.max_wn
        else:
            self.max_wn_plot = max_wn_plot

        if (self.max_wn_plot>self.max_wn):
            print("Warning: max_wn_plot is higher than the values used in the analysis. \n\
            Changing the parameter for the one used in the analysis.")
            self.max_wn_plot = self.max_freq
        if (self.max_freq_plot>self.max_freq):
            print("Warning: max_wn_plot is higher than the values used in the analysis. \n\
            Changing the parameter for the one used in the analysis.")
            self.max_freq_plot = self.max_freq

        psumanti_log = self.wk_spectra['psumanti_log']
        psumsym_log = self.wk_spectra['psumsym_log']
        psumb_log = self.wk_spectra['psumb_log']
        wavefft = self.wk_spectra['wavefft']
        freqfft = self.wk_spectra['freqfft']

        indwave = np.where(np.logical_and(wavefft>=-self.max_wn_plot-0.1,wavefft<=self.max_wn_plot+0.1))[0]
        indfreq = np.where(np.logical_and(freqfft>0,freqfft<=self.max_freq_plot+0.1))[0]

        waveplt = wavefft[indwave]
        freqplt = freqfft[indfreq]

        X,Y = np.meshgrid(waveplt,freqplt)

        contour_range=np.arange(contour_range[0],contour_range[1],contour_range[2])

        powerplt = psumanti_log[indfreq,:]
        powerplt = powerplt[:,indwave]
        figA,axA = raw_plotter(X,Y,powerplt,self.max_wn_plot,self.max_freq_plot,\
                              figsize,contour_range,cmap,text_size,freq_lines,cpd_lines)
        self.plots['antisymmetric_raw_log'] = figA

        powerplt = psumsym_log[indfreq,:]
        powerplt = powerplt[:,indwave]
        figS,axS = raw_plotter(X,Y,powerplt,self.max_wn_plot,self.max_freq_plot,\
                              figsize,contour_range,cmap,text_size,freq_lines,cpd_lines)
        self.plots['symmetric_raw_log'] = figS

        powerplt = psumb_log[indfreq,:]
        powerplt = powerplt[:,indwave]
        figB,axB = raw_plotter(X,Y,powerplt,self.max_wn_plot,self.max_freq_plot,\
                              figsize,contour_range,cmap,text_size,freq_lines,cpd_lines)
        self.plots['background_log'] = figB

        return None

    def plot_dual_raw(self,figsize=(18,8),text_size=12,contour_range=(-1.9,-0.15,0.1),\
                        cmap='Reds',max_freq_plot=None,max_wn_plot=None,freq_lines=True,\
                        cpd_lines=[3,6,30]):

            if (self.plots_available == "No"):
                self.plots_available = "Yes"
                self.plots = {}

            if max_freq_plot is None:
                self.max_freq_plot = self.max_freq
            else:
                self.max_freq_plot = max_freq_plot
            if max_wn_plot is None:
                self.max_wn_plot = self.max_wn
            else:
                self.max_wn_plot = max_wn_plot

            if (self.max_wn_plot>self.max_wn):
                print("Warning: max_wn_plot is higher than the values used in the analysis. \n\
                Changing the parameter for the one used in the analysis.")
                self.max_wn_plot = self.max_freq
            if (self.max_freq_plot>self.max_freq):
                print("Warning: max_wn_plot is higher than the values used in the analysis. \n\
                Changing the parameter for the one used in the analysis.")
                self.max_freq_plot = self.max_freq

            psumanti_log = self.wk_spectra['psumanti_log']
            psumsym_log = self.wk_spectra['psumsym_log']
            wavefft = self.wk_spectra['wavefft']
            freqfft = self.wk_spectra['freqfft']

            indwave = np.where(np.logical_and(wavefft>=-self.max_wn_plot-0.1,wavefft<=self.max_wn_plot+0.1))[0]
            indfreq = np.where(np.logical_and(freqfft>0,freqfft<=self.max_freq_plot+0.1))[0]

            waveplt = wavefft[indwave]
            freqplt = freqfft[indfreq]

            X,Y = np.meshgrid(waveplt,freqplt)

            contour_range=np.arange(contour_range[0],contour_range[1],contour_range[2])

            figAS, axAS = plt.subplots(1,2,figsize=figsize)

            powerplt = psumanti_log[indfreq,:]
            powerplt = powerplt[:,indwave]

            cset_0  = axAS[0].contourf(X,Y,powerplt,contour_range,extend='both',cmap=cmap)
            cset1_0 = axAS[0].contour(X,Y,powerplt,contour_range,colors='k')

            powerplt = psumsym_log[indfreq,:]
            powerplt = powerplt[:,indwave]

            cset_1  = axAS[1].contourf(X,Y,powerplt,contour_range,extend='both',cmap=cmap)
            cset1_1 = axAS[1].contour(X,Y,powerplt,contour_range,colors='k')

            for ax in axAS:
                ax.xaxis.set_tick_params(labelsize=text_size)
                ax.yaxis.set_tick_params(labelsize=text_size)

                ax.axvline(x=0, color='k', linestyle='--')

                ax.set_xlabel('Zonal Wavenumber',size=text_size,fontweight='bold')
                ax.set_ylabel('Frequency (CPD)',size=text_size,fontweight='bold')
                ax.text(self.max_wn_plot-2*0.25*self.max_wn_plot,-0.01,'EASTWARD',fontweight='bold',fontsize=text_size-2)
                ax.text(-self.max_wn_plot+0.25*self.max_wn_plot,-0.01,'WESTWARD',fontweight='bold',fontsize=text_size-2)
                ax.set_xlim((-self.max_wn_plot,self.max_wn_plot))
                ax.set_ylim((0.02,self.max_freq_plot))

                if freq_lines:
                    for d in cpd_lines:
                        if ((1./d)<=self.max_freq_plot):
                            ax.axhline(y=1./d,color='k', linestyle='--')
                            ax.text(-self.max_wn_plot+0.2,(1./d+0.01),str(d)+' days',\
                            size=text_size,bbox={'facecolor':'white','alpha':0.9,\
                            'edgecolor':'none'})

            self.plots['anti_sym_raw_log'] = figAS

            return None

    def plot_background_removed(self,figsize=(10,8),text_size=12,contour_range=(0,2,0.1),min_contour_range_lines=1.1,\
                                matsuno_lines=True,he=[12,25,50],meridional_modes=[1],cmap='coolwarm',max_freq_plot=None,\
                                max_wn_plot=None,freq_lines=True,cpd_lines=[3,6,30],labels=False):

        contour_range = np.arange(contour_range[0],contour_range[1],contour_range[2])

        contour_range_lines = contour_range[contour_range>=min_contour_range_lines]

        if contour_range_lines==[]:
            print('Warning: The selected minimum value for contour lines is higher\n\
            than the maximum contour value.  The value is changed to the minimum contour.')
            contour_range_lines = contour_range

        if (self.plots_available == "No"):
            self.plots_available = "Yes"
            self.plots = {}

        if max_freq_plot is None:
            self.max_freq_plot = self.max_freq
        else:
            self.max_freq_plot = max_freq_plot
        if max_wn_plot is None:
            self.max_wn_plot = self.max_wn
        else:
            self.max_wn_plot = max_wn_plot

        if (self.max_wn_plot>self.max_wn):
            print("Warning: max_wn_plot is higher than the values used in the analysis. \n\
            Changing the parameter for the one used in the analysis.")
            self.max_wn_plot = self.max_freq
        if (self.max_freq_plot>self.max_freq):
            print("Warning: max_wn_plot is higher than the values used in the analysis. \n\
            Changing the parameter for the one used in the analysis.")
            self.max_freq_plot = self.max_freq

        psumanti_r = self.wk_spectra['psumanti_r']
        psumsym_r = self.wk_spectra['psumsym_r']
        wavefft = self.wk_spectra['wavefft']
        freqfft = self.wk_spectra['freqfft']

        indwave = np.where(np.logical_and(wavefft>=-self.max_wn_plot-0.1,wavefft<=self.max_wn_plot+0.1))[0]
        indfreq = np.where(np.logical_and(freqfft>0,freqfft<=self.max_freq_plot+0.1))[0]

        waveplt = wavefft[indwave]
        freqplt = freqfft[indfreq]

        X,Y = np.meshgrid(waveplt,freqplt)



        powerplt = psumanti_r[indfreq,:]
        powerplt = powerplt[:,indwave]
        figA,axA = background_removed_plotter(X,Y,powerplt,self.max_wn_plot,self.max_freq_plot,figsize,\
                                              contour_range,contour_range_lines,cmap,text_size,freq_lines,cpd_lines,matsuno_lines,he,\
                                              meridional_modes,False,labels)

        self.plots['antisymmetric_background_rem'] = figA


        powerplt = psumsym_r[indfreq,:]
        powerplt = powerplt[:,indwave]
        figS,axS = background_removed_plotter(X,Y,powerplt,self.max_wn_plot,self.max_freq_plot,figsize,\
                                              contour_range,contour_range_lines,cmap,text_size,freq_lines,cpd_lines,matsuno_lines,he,\
                                              meridional_modes,True,labels)

        self.plots['symmetric_background_rem'] = figS

        return None

    def plot_dual_background_removed(self,figsize=(18,8),text_size=12,contour_range=(0,2,0.1),min_contour_range_lines=1.1,\
                                matsuno_lines=True,he=[12,25,50],meridional_modes=[1],cmap='coolwarm',max_freq_plot=None,\
                                max_wn_plot=None,freq_lines=True,cpd_lines=[3,6,30],labels=False):
            contour_range = np.arange(contour_range[0],contour_range[1],contour_range[2])

            contour_range_lines = contour_range[contour_range>=min_contour_range_lines]

            if contour_range_lines==[]:
                print('Warning: The selected minimum value for contour lines is higher\n\
                than the maximum contour value.  The value is changed to the minimum contour.')
                contour_range_lines = contour_range

            if (self.plots_available == "No"):
                self.plots_available = "Yes"
                self.plots = {}

            if max_freq_plot is None:
                self.max_freq_plot = self.max_freq
            else:
                self.max_freq_plot = max_freq_plot
            if max_wn_plot is None:
                self.max_wn_plot = self.max_wn
            else:
                self.max_wn_plot = max_wn_plot

            if (self.max_wn_plot>self.max_wn):
                print("Warning: max_wn_plot is higher than the values used in the analysis. \n\
                Changing the parameter for the one used in the analysis.")
                self.max_wn_plot = self.max_freq
            if (self.max_freq_plot>self.max_freq):
                print("Warning: max_wn_plot is higher than the values used in the analysis. \n\
                Changing the parameter for the one used in the analysis.")
                self.max_freq_plot = self.max_freq

            psumanti_r = self.wk_spectra['psumanti_r']
            psumsym_r = self.wk_spectra['psumsym_r']
            wavefft = self.wk_spectra['wavefft']
            freqfft = self.wk_spectra['freqfft']

            indwave = np.where(np.logical_and(wavefft>=-self.max_wn_plot-0.1,wavefft<=self.max_wn_plot+0.1))[0]
            indfreq = np.where(np.logical_and(freqfft>0,freqfft<=self.max_freq_plot+0.1))[0]

            waveplt = wavefft[indwave]
            freqplt = freqfft[indfreq]

            X,Y = np.meshgrid(waveplt,freqplt)

            figAS, axAS = plt.subplots(1,2,figsize=figsize)

            powerplt = psumanti_r[indfreq,:]
            powerplt = powerplt[:,indwave]

            cset_0  = axAS[0].contourf(X,Y,powerplt,contour_range,extend='both',cmap=cmap)
            cset1_0 = axAS[0].contour(X,Y,powerplt,contour_range_lines,colors='k')

            powerplt = psumsym_r[indfreq,:]
            powerplt = powerplt[:,indwave]

            cset_1  = axAS[1].contourf(X,Y,powerplt,contour_range,extend='both',cmap=cmap)
            cset1_1 = axAS[1].contour(X,Y,powerplt,contour_range_lines,colors='k')


            for ax in axAS:
                ax.xaxis.set_tick_params(labelsize=text_size)
                ax.yaxis.set_tick_params(labelsize=text_size)

                ax.axvline(x=0, color='k', linestyle='--')

                ax.set_xlabel('Zonal Wavenumber',size=text_size,fontweight='bold')
                ax.set_ylabel('Frequency (CPD)',size=text_size,fontweight='bold')
                ax.text(self.max_wn_plot-2*0.25*self.max_wn_plot,-0.01,'EASTWARD',fontweight='bold',fontsize=text_size-2)
                ax.text(-self.max_wn_plot+0.25*self.max_wn_plot,-0.01,'WESTWARD',fontweight='bold',fontsize=text_size-2)
                ax.set_xlim((-self.max_wn_plot,self.max_wn_plot))
                ax.set_ylim((0.02,self.max_freq_plot))

                if freq_lines:
                    for d in cpd_lines:
                        if ((1./d)<=self.max_freq_plot):
                            ax.axhline(y=1./d,color='k', linestyle='--')
                            ax.text(-self.max_wn_plot+0.2,(1./d+0.01),str(d)+' days',\
                            size=text_size,bbox={'facecolor':'white','alpha':0.9,\
                            'edgecolor':'none'})


            if matsuno_lines:
                matsuno_modes = mp.matsuno_modes_wk(he=he,n=meridional_modes,max_wn=self.max_wn_plot)

                for key in matsuno_modes:
                    axAS[1].plot(matsuno_modes[key]['Kelvin(he={}m)'.format(key)],color='k',linestyle='--')
                    axAS[1].plot(matsuno_modes[key]['ER(n=1,he={}m)'.format(key)],color='k',linestyle='--')
                    axAS[1].plot(matsuno_modes[key]['EIG(n=1,he={}m)'.format(key)],color='k',linestyle='--')
                    axAS[1].plot(matsuno_modes[key]['WIG(n=1,he={}m)'.format(key)],color='k',linestyle='--')
                    axAS[0].plot(matsuno_modes[key]['MRG(he={}m)'.format(key)],color='k',linestyle='--')
                    axAS[0].plot(matsuno_modes[key]['EIG(n=0,he={}m)'.format(key)],color='k',linestyle='--')


                if labels:
                    key = list(matsuno_modes.keys())[len(list(matsuno_modes.keys()))//2]
                    wn = matsuno_modes[key].index.values

                    # Print Kelvin Label
                    i = int((len(wn)/2)+0.3*(len(wn)/2))
                    i, = np.where(wn == wn[i])[0]
                    axAS[1].text(wn[i]-1,matsuno_modes[key]['Kelvin(he={}m)'.format(key)].iloc[i],'Kelvin', \
                    bbox={'facecolor':'white','alpha':0.9,'edgecolor':'none'},fontsize=text_size+4)

                    # Print ER Label
                    i = int(0.7*(len(wn)/2))
                    i = np.where(wn == wn[i])[0]
                    axAS[1].text(wn[i]-1,matsuno_modes[key]['ER(n=1,he={}m)'.format(key)].iloc[i]+0.01,'ER', \
                    bbox={'facecolor':'white','alpha':0.9,'edgecolor':'none'},fontsize=text_size+1)

                    # Print EIG(n=0) Label
                    i = int((len(wn)/2)+0.1*(len(wn)/2))
                    i, = np.where(wn == wn[i])[0]
                    axAS[0].text(wn[i]-1,matsuno_modes[key]['EIG(n=0,he={}m)'.format(key)].iloc[i],'EIG(n=0)', \
                    bbox={'facecolor':'white','alpha':0.9,'edgecolor':'none'},fontsize=text_size+1)

                    # Print MRG Label
                    i = int(0.7*(len(wn)/2))
                    i, = np.where(wn == wn[i])[0]
                    axAS[0].text(wn[i]-1,matsuno_modes[key]['MRG(he={}m)'.format(key)].iloc[i],'MRG', \
                    bbox={'facecolor':'white','alpha':0.9,'edgecolor':'none'},fontsize=text_size+1)

                    key2 = list(matsuno_modes.keys())[0]
                    wn2 = matsuno_modes[key].index.values

                    # Print EIG Label
                    i = int((len(wn2)/2)+0.3*(len(wn2)/2))
                    i, = np.where(wn2 == wn2[i])[0]
                    axAS[1].text(wn2[i]-1,matsuno_modes[key2]['EIG(n=1,he={}m)'.format(key2)].iloc[i],'EIG', \
                    bbox={'facecolor':'white','alpha':0.9,'edgecolor':'none'},fontsize=text_size+1)

                    # Print WIG Label
                    i = int(0.55*(len(wn2)/2))
                    i, = np.where(wn2 == wn2[i])[0]
                    axAS[1].text(wn2[i]-1,matsuno_modes[key2]['WIG(n=1,he={}m)'.format(key2)].iloc[i],'WIG', \
                    bbox={'facecolor':'white','alpha':0.9,'edgecolor':'none'},fontsize=text_size+1)

            self.plots['anti_sym_background_rem'] = figAS

            return None

    def export_data(self,name=None):
        if name is None:
            if hasattr(self, 'name'):
                name = self.name
            else:
                name = str(self.varname)

        name = name+'.wk_spectra.npy'
        np.save(name,self.wk_spectra)
        print('The analysis is saved in the file {}'.format(name))

        return None

    def import_analysis(self,file):
        if (self.data_status=="Empty"):
            self.data_status = "An analysis file has been imported"
            self.analysis_status = "Imported"
            self.wk_spectra = np.load(file).item()
            self.spd = self.wk_spectra['spd']
            self.nDayWin = self.wk_spectra['nDayWin']
            self.nDaySkip = self.wk_spectra['nDaySkip']
            self.max_freq = self.wk_spectra['max_freq']
            self.max_wn = self.wk_spectra['max_wn']
            self.varname = self.wk_spectra['varname']
            print('The analysis stored in {} has been imported.'.format(file))
        else:
            raise InputError("You have already loaded some data")
        return None

    def save_figs(self,name=None,ext='png'):
        if (self.plots_available == "Yes"):
            g_ext = ['png','pdf','eps']
            if ext not in g_ext:
                print('The file extension {} is not supported. Using png instead.'\
                .format(str(ext)))
                ext = 'png'
            if name is None:
                if hasattr(self, 'name'):
                    name = self.name
                else:
                    name = str(self.varname)
            dirname = name+'_plots'
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            for key in self.plots:
                self.plots[key].savefig(dirname+'/'+name+'_'+key+'.'+ext)
        else:
            raise InputError("You don't have plots to save")

        return None
