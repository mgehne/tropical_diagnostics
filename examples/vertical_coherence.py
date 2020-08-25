"""
Example script to compute vertical coherence profiles.

This script assumes that the filtered precipitation file already exists. If not, the user needs to
run filter_CCEW.py to generate the filtered data. The user needs to change the input paths and
filenames and the output location. Variable names depend on the input data and need to be specified
by the user as well.
"""
from tropical_diagnostics.diagnostics import vertical_coherence as vc


var1 = "precip"  # variable name of data in the precipitation file
var2 = "shum"    # variable name of data in the second file

source1 = "TRMM"  # ERAI, ERA5, TRMM
source2 = "ERA5"

wave1 = "MJO"  # "kelvin", "MRG", "ER", "MJO", "IG0"

RES = "2p5"  # spatial resolution of the data
spd = 1  # data is spd x daily

# var2 is read at all these levels
level2 = [1000,975,950,925,900,875,850,825,800,775,750,700,650,600,550,500,
          450,400,350,300,250,225,200,175,150,125,100]

# first and last date format: yyyymmddhh
datemin = 2007010100
datemax = 2010123100
yearmin = datemin/1000000
yearmax = datemax/1000000

# significance level for the coherence plots
sigstr = 99.
siglev = sigstr/100

# latBound
latN = 20
latS = -latN

# Wheeler - Kiladis[WK] temporal window length(days)
nDayWin = 128 * spd
nDaySkip = -32 * spd

# input file names
filebase = 'precip.trmm.'+str(spd)+'x.'+RES+'.v7a.fillmiss.comp.1998-201806'
pathin = '/data/mgehne/Precip/MetricsObs/CCEWfilter/'
pathout = '/data/mgehne/VerticalCoherence/'
plotpath = "~/Projects/Diagnostics/Plots/VerticalCoherence/"
outfile = "CoherenceVertical_python_"+RES+"_"+str(spd)+"x_"+source1+var1+wave1+"_"+source2+var2+"_"\
          + str(datemin)+"-"+str(datemax)+"_"+str(latS)+"-"+str(latN)+"_sigMask"
outfileSpectra = "CoherenceVertical_SpaceTime_python_"+RES+"_"+str(spd)+"x_"+source1+var1+wave1+"_"+source2+var2+"_"\
          + str(datemin)+"-"+str(datemax)+"_"+str(latS)+"-"+str(latN)+"_sigMask"


# read data1

# remove annual cycle from data1 - not necessary unless using unfiltered data

# read data2

# put this next part into a function
CohAvg, CohMask, CohMat = vc.vertical_coherence_comp(data1, data2, levels, nDayWin, nDaySkip, spd)

# save data to file

# plot vertical coherence profile
