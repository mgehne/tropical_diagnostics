import Ngl as ngl
import numpy as np
from tropical_diagnostics.diagnostics import spacetime_plot as stp
import string

from netCDF4 import Dataset

pathdata = './data/'
plotpath = './plots/'

# plot layout parameters
flim = 0.5            # maximum frequency in cpd for plotting
nWavePlt = 20         # maximum wavenumber for plotting
contourmin = 0.05     # contour minimum
contourmax = 0.55     # contour maximum
contourspace = 0.05   # contour spacing
N = [1,2]             # wave modes for plotting
source = "ERAI"
var1 = "P"
var2 = "D850"

symmetry = ("symm","asymm","latband")
nplot = len(symmetry)
pp = 0

while pp<nplot:

# read data from file
    fin = Dataset(pathdata+'SpaceTimeSpectra_'+symmetry[pp]+'_2spd.nc',"r")
    STC  = fin['STC'][:,:,:]
    wnum = fin['wnum']
    freq = fin['freq']

    ifreq = np.where((freq[:]>=0) & (freq[:]<=flim))
    iwave = np.where(abs(wnum[:])<=nWavePlt)

    STC[:,freq[:]==0,:] = 0.
    STC = STC[:,:,iwave]
    STC = STC[:,ifreq,:]
    coh2 = np.squeeze(STC[4,:,:])
    phs1 = np.squeeze(STC[6,:,:])
    phs2 = np.squeeze(STC[7,:,:])
    pow1 = np.squeeze(STC[0,:,:])
    pow2 = np.squeeze(STC[1,:,:])
    pow1[pow1<=0] = np.nan
    pow2[pow2<=0] = np.nan

    if pp==0:
        Coh2 = np.empty([nplot,len(freq[ifreq]),len(wnum[iwave])])
        Phs1 = np.empty([nplot,len(freq[ifreq]),len(wnum[iwave])])
        Phs2 = np.empty([nplot,len(freq[ifreq]),len(wnum[iwave])])
        Pow1 = np.empty([nplot,len(freq[ifreq]),len(wnum[iwave])])
        Pow2 = np.empty([nplot,len(freq[ifreq]),len(wnum[iwave])])
        

    Coh2[pp,:,:] = coh2
    Phs1[pp,:,:] = phs1
    Phs2[pp,:,:] = phs2
    Pow1[pp,:,:] = np.log10(pow1)
    Pow2[pp,:,:] = np.log10(pow2)


    phstmp = Phs1
    phstmp = np.square(Phs1) + np.square(Phs2)
    phstmp = np.where(phstmp==0,np.nan,phstmp)
    scl_one = np.sqrt(1/phstmp)
    Phs1    = scl_one*Phs1
    Phs2    = scl_one*Phs2

    pp += 1

# plot coherence
stp.plot_coherence(Coh2,Phs1,Phs2,symmetry,source,var1,var2,plotpath,flim,20,contourmin,contourmax,contourspace,nplot,N)

exit()
