"""
Scripts for generating Hovmoeller diagrams.

"""

import numpy as np
from datetime import datetime
from time import strftime
from netCDF4 import num2date, date2num, date2index
import Ngl as ngl
import string

def hov_resources(time,lon,pltvarname,wks,FillMode="AreaFill",cmin=[],cmax=[],cspc=[]):
    
    res = ngl.Resources()

    if ((not cmin) or (not cmax) or (not cspc)):
        if pltvarname=='precip':
            clevels = [0.2, 0.4, 0.6, 0.8, 1.1, 1.5]
            cmap = ["white","black","white","lightskyblue","dodgerblue","blue2","slateblue1","mediumpurple1","purple2"]
            ngl.define_colormap(wks,cmap)
        if pltvarname=='uwnd':
            clevels = [-30, -20, -14, -10, -7, -4, -2, -1, 1, 2, 4, 7, 10, 14, 20, 30]
            fillpalette = "BlueWhiteOrangeRed"
            res.cnFillPalette = fillpalette
        if pltvarname=='vwnd':
            clevels = [-30, -20, -14, -10, -7, -4, -2, -1, 1, 2, 4, 7, 10, 14, 20, 30]
            fillpalette = "BlueWhiteOrangeRed"
            res.cnFillPalette = fillpalette
        if pltvarname=='div':
            clevels = [-0.00002,-0.00001,-0.000005,-0.000001,0.000001,0.000005,0.00001,0.00002]
            fillpalette = "BlueWhiteOrangeRed"
            res.cnFillPalette = fillpalette
        if pltvarname=='olr':
            clevels = [160, 170, 190, 210, 230, 260]
            cmap = ["white","black","purple2","mediumpurple1","slateblue1","blue2","dodgerblue","lightskyblue","white"]
            ngl.define_colormap(wks,cmap)    
    else:
        clevels    = np.arange(cmin,cmax,cspc)
        fillpalette = "BlueWhiteOrangeRed"
        res.cnFillPalette = fillpalette
        
    ts = (time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 'h')
    date = num2date(ts,'hours since 1970-01-01T00:00:00Z')
    timestr = [i.strftime("%Y-%m-%d %H:%M") for i in date]

    res.nglDraw  = False
    res.nglFrame = False
    res.cnLinesOn = False
    res.cnFillOn = True
    res.cnFillMode = FillMode
    res.cnLineLabelsOn = False
    res.cnInfoLabelOn = False
    res.lbLabelBarOn = False
    
    res.cnLevelSelectionMode = 'ExplicitLevels'
    res.cnLevels             = clevels
    res.tiYAxisString = ""
    res.tiXAxisString = "longitude"
    res.trYReverse    = "True"
    res.tmYLMode   = 'Explicit'
    res.tmYLValues = np.arange(0,len(timestr)-1,10)
    res.tmYLLabels = timestr[::10]
    res.tmYLMinorValues = np.arange(0,len(timestr)-1,1)
    res.tmXBMode   = 'Explicit'
    res.tmXBValues = np.arange(0,len(lon)-1,30/360*len(lon))
    res.tmXBLabels = np.arange(0,lon.max(),(lon.max()+lon[1])*30/360)
    res.tmXBMinorValues = np.arange(0,len(lon)-1,15/360*len(lon))

    return res

def panel_resources(res,nplot=4,latS=None,latN=None,units=[]):
    if latS<0:
        hemS = 'S'
        latS = -latS
    else:
        hemS = 'N'
    if latN<0:
        hemN = 'S'
        latN = -latN
    else:
        hemN = 'N'
    resP  = ngl.Resources()
    resP.nglFrame            = True
    resP.nglMaximize         = True
    resP.nglPanelLabelBar    = True
    #resP.nglPanelRight = 0.9
    #resP.nglPanelLeft = 0.15
    #resP.nglPanelBottom = 0.05
    resP.lbOrientation     = "vertical"
    resP.lbTitleString     = units
    resP.lbLabelStrings    = ["{0:0.2f}".format(i) for i in res.cnLevels]
    resP.nglPanelLabelBarLabelFontHeightF = 0.02
    resP.nglPanelLabelBarHeightF = 0.37
    resP.nglPanelLabelBarParallelPosF = 0.025
    resP.nglPanelFigureStrings = [str(latS)+hemS+" - "+str(latN)+hemN]
    resP.nglPanelFigureStringsJust = "TopRight"

    return resP


def hovmoeller(A,lon,time,datestrt,datelast,spd,source,pltvarname,plotpath,latS,latN,cmin=[],cmax=[],cspc=[]):
    FillMode = "AreaFill"   
   
    # open plot workstation
    wkstype = "png"
    wks = ngl.open_wks(wkstype,plotpath+"Hovmoeller_"+source+pltvarname+"_"+str(datestrt)+"-"+str(datelast))
    plots = []

    # plot resources
    res  = hov_resources(time,lon,pltvarname,wks,FillMode,cmin,cmax,cspc)
    resP = panel_resources(res,1,latS,latN,A.attrs['units'])
    
    # plot hovmoeller
    plot = ngl.contour(wks,A.values,res)
    plots.append(plot)
    
    # panel plots    
    ngl.panel(wks,plots,[1,1],resP)

    ngl.end()
