"""
Scripts for generating Hovmoeller diagrams.

"""

import numpy as np
import plotly.graph_objects as go
from netCDF4 import num2date


def hov_resources(time, lon, pltvarname, FillMode="AreaFill", cmin=[], cmax=[], cspc=[]):
    if pltvarname == 'precip':
        clevels = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        # cmap = ["white", "lightskyblue", "dodgerblue", "blue2", "slateblue1", "mediumpurple1", "purple2"]
        cmap_rgb = [[0, "rgb(255, 255, 255)"], [0.14, "rgb(255, 255, 255)"],
                    [0.14, "rgb(135, 206, 250)"], [0.28, "rgb(135, 206, 250)"],
                    [0.28, "rgb(30, 144, 255)"], [0.42, "rgb(30, 144, 255)"],
                    [0.42, "rgb(0, 0, 238)"], [0.56, "rgb(0, 0, 238)"],
                    [0.56, "rgb(131, 111, 255)"], [0.71, "rgb(131, 111, 255)"],
                    [0.71, "rgb(171, 130, 255)"], [0.85, "rgb(171, 130, 255)"],
                    [0.85, "rgb(145, 44, 238)"], [1, "rgb(145, 44, 238)"]]
    if pltvarname == 'uwnd':
        clevels = [-30, -20, -14, -10, -7, -4, -2, -1, 1, 2, 4, 7, 10, 14, 20, 30]
        fillpalette = "BlueWhiteOrangeRed"
    if pltvarname == 'vwnd':
        clevels = [-30, -20, -14, -10, -7, -4, -2, -1, 1, 2, 4, 7, 10, 14, 20, 30]
        fillpalette = "BlueWhiteOrangeRed"
    if pltvarname == 'div':
        clevels = [-0.00002, -0.00001, -0.000005, -0.000001, 0.000001, 0.000005, 0.00001, 0.00002]
        fillpalette = "BlueWhiteOrangeRed"
    if pltvarname == 'olr':
        clevels = [160, 170, 190, 210, 230, 260]
        # cmap = ["purple2", "mediumpurple1", "slateblue1", "blue2", "dodgerblue", "lightskyblue", "white"]
        cmap_rgb = [[145, 44, 238], [171, 130, 255], [131, 111, 255], [0, 0, 238],
                    [30, 144, 255], [135, 206, 250], [255, 255, 255]]

    ts = (time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 'h')
    date = num2date(ts, 'hours since 1970-01-01T00:00:00Z')
    timestr = [i.strftime("%Y-%m-%d %H:%M") for i in date]


    return cmap_rgb, timestr


"""
def panel_resources(res, nplot=4, latS=None, latN=None, units=[]):
    if latS < 0:
        hemS = 'S'
        latS = -latS
    else:
        hemS = 'N'
    if latN < 0:
        hemN = 'S'
        latN = -latN
    else:
        hemN = 'N'
    resP = ngl.Resources()
    resP.nglFrame = True
    resP.nglMaximize = True
    resP.nglPanelLabelBar = True
    resP.lbOrientation = "vertical"
    resP.lbTitleString = units
    resP.lbLabelStrings = ["{0:0.2f}".format(i) for i in res.cnLevels]
    resP.nglPanelLabelBarLabelFontHeightF = 0.02
    resP.nglPanelLabelBarHeightF = 0.37
    resP.nglPanelLabelBarParallelPosF = 0.025
    resP.nglPanelFigureStrings = [str(latS) + hemS + " - " + str(latN) + hemN]
    resP.nglPanelFigureStringsJust = "TopRight"

    return resP
"""


def hovmoeller(data, lon, time, datestrt, datelast, spd, source, pltvarname, plotpath, latS, latN, cmin=[], cmax=[],
               cspc=[]):
    # open plot workstation
    plttype = "png"
    plotname = plotpath + "Hovmoeller_" + source + pltvarname + "_" + str(datestrt) + "-" + str(
        datelast) + "." + plttype

    # plot resources
    cmap_rgb, timestr = hov_resources(time, lon, pltvarname, cmin, cmax, cspc)
    # resP = panel_resources(res, 1, latS, latN, data.attrs['units'])

    # plot hovmoeller
    fig = go.Figure()

    fig.add_trace(go.Contour(
        z=data.values,
        x=lon,
        y=timestr,
        colorscale=cmap_rgb,
        contours=dict(start=cmin, end=cmax, size=cspc,
                      showlines=False)
    ))

    fig.update_layout(
        title=source + " " + pltvarname,
        width=600,
        height=900)

    fig.update_xaxes(ticks="inside", tick0=0, dtick=30, title_text='longitude')
    fig.update_yaxes(autorange="reversed", ticks="inside", nticks=11)

    fig.write_image(plotname)

    return
