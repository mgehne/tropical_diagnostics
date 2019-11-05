import Ngl as ngl
import numpy as np
from diagnostics import  spacetime_plot as stp
import string

from netCDF4 import Dataset

pathdata = './data/'
plotpath = './plots/'

# plot layout parameters
flim = 0.5            # maximum frequency in cpd for plotting
nWavePlt = 15         # maximum wavenumber for plotting
contourmin = -5     # contour minimum
contourmax = -1.      # contour maximum
contourspace = 0.5    # contour spacing
N = [1,2]             # wave modes for plotting
source = "KWmodel"
var1 = "U"
spd = 4

symmetry = ("symm","asymm","latband")
nplot = len(symmetry)
pp = 0

while pp<nplot:

# read data from file
    fin = Dataset(pathdata+'SpaceTimeSpectra_'+symmetry[pp]+'_'+str(spd)+'spd.nc',"r")
    STC = fin['STC'][:,:,:]
    wnum = fin['wnum']
    freq = fin['freq']

    ifreq = np.where((freq[:]>=0) & (freq[:]<=flim))
    iwave = np.where(abs(wnum[:])<=nWavePlt)

    STC[:,freq[:]==0,:] = 0.
    STC = STC[:,:,iwave]
    STC = STC[:,ifreq,:]
    pow1 = np.squeeze(STC[0,:,:])
    pow2 = np.squeeze(STC[1,:,:])
    pow1[pow1<=0] = np.nan
    pow2[pow2<=0] = np.nan

    if pp==0:
        Pow1 = np.empty([nplot,len(freq[ifreq]),len(wnum[iwave])])
        Pow2 = np.empty([nplot,len(freq[ifreq]),len(wnum[iwave])])
        
    Pow1[pp,:,:] = np.log10(pow1)
    Pow2[pp,:,:] = np.log10(pow2)

    pp += 1


stp.plot_power(Pow1,symmetry,source,var1,plotpath,flim,20,contourmin,contourmax,contourspace,nplot,N)
exit()
