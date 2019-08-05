from netCDF4 import Dataset
import numpy as np

def save_Spectra(STCin,freq_in,wnum_in,filename,filepath,opt=False):
    nc = Dataset(filepath+filename+'.nc', 'w', format='NETCDF4')

    nvar, nfrq, nwave = STCin.shape
# dimensions
    nc.createDimension('freq', nfrq)
    nc.createDimension('wnum', nwave)
    nc.createDimension('var', nvar)
    
    
# variables
    freq = nc.createVariable('freq', 'double', ('freq',))
    wnum = nc.createVariable('wnum', 'int', ('wnum',))
    var  = nc.createVariable('var', 'int', ('var',))
    STC = nc.createVariable('STC', 'double', ('var', 'freq', 'wnum',))

# attributes
    STC.varnames = ['PX','PY','CXY','QXY','COH2','PHA','V1','V2']
    STC.long_name = "Space time spectra"
    freq.units = "cpd"
    freq.long_name = "frequency"
    wnum.units = ""
    wnum.long_name = "zonal wavenumber"
    var.long_name = "variable number"
    
# data
    var[:]      = np.linspace(0, nvar-1, nvar)
    freq[:]     = freq_in
    wnum[:]     = wnum_in
    STC[:,:,:]  = STCin

    nc.close()
