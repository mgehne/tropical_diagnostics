import numpy as np
"""
local scripts, if loading from a different directory include that with a '.' between
directory name and script name
"""
from utils.readdata import readdata       
import utils.hovmoeller as hov

plotpath = '~/Projects/Diagnostics/Python/plots/'

"""
Parameters to set for the Hovmoeller diagrams.
"""
spd = 2              # number of obs per day
source = "OLR"      # data source

var1 = "olr"       # variable to read
lev1 = -1             # vertical levelm set to -1 if single level variable
datestrt = 2016010100 # plot start date, format: yyyymmddhh 
datelast = 2016013100 # plot end date, format: yyyymmddhh

latN =  5.            # maximum latitude for the average 
latS = -5.            # minimum latitude for the average 

# plotting parameters
pltvarname = "olr"    # variable name for plotting, assumes units mm/h (precip), m/s (winds), 1/s (divergence)
#contourmin = 0.1      # contour minimum, optional
#contourmax = 1.5      # contour maximum, optional
#contourspace = 0.1    # contour spacing, optional


print("reading data from file:")
A, latA, lonA, timeA = readdata(var1,lev1,source,"",datestrt,datelast,spd)
units = A.units

print("extracting latitude bands:")
A    = A[:,(latS<=latA) & (latA<=latN),:]
latA = latA[(latS<=latA) & (latA<=latN)]

    
print("average over latitude band:")
A = np.average(A,axis=1)
A.units = units
if source=='ERAI' and var1=="precip":
    A.units = 'mm/h'

print("plot hovmoeller diagram:")
hov.hovmoeller(A,lonA,timeA,datestrt,datelast,spd,source,pltvarname,plotpath,latS,latN)#,contourmin,contourmax,contourspace)
