"""
Vertical coherence profile plots using plotly module. Use spacetime_plot.py to
plot full spectra
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from kaleido.scopes.plotly import PlotlyScope

def get_latstring(lats, latn):
    """
    Generate string describing the latitude band averaged over.
    :param lats: southern latitude limit of the average
    :type lats: float
    :param latn: northern latitude limit of the average
    :type latn: float
    :return: latstr
    :rtype: str
    """
    if lats < 0:
        hems = 'S'
        lats = -lats
    else:
        hems = 'N'
    if latn < 0:
        hemn = 'S'
        latn = -latn
    else:
        hemn = 'N'

    latstr = str(lats) + hems + " - " + str(latn) + hemn

    return latstr


def plot_vertcoh(coh, px, py, levels, labels, titlestr, plotname, plotpath, lats, latn, xlim=[0, 0.5]):
    """
    Plot averaged coherence and phase values by level.
    :param coh: Averaged coherence values. nvar x nlevels
    :param px: Averaged phase angle (x-component) values. nvar x nlevels
    :param py: Averaged phase angle (y-component) values. nvar x nlevels
    :param levels: Vertical level coordinate.
    :param labels: Labels for each variable, should include variable and symm/ anti-symm tag.
    :param titlestr: Title for the plot, should include CCEW/ MJO tag and variable name for var1 (usually precip).
    :param plotname: Name for the plot to be saved under.
    :param plotpath: Path for the plot to be saved at.
    :param lats: Southern latitude value the spectra were averaged over.
    :param latn: Northern latitude value the spectra were averaged over.
    :param xlim: optional parameter specifying the maximum coherence value on the x-axis
    :return:
    """

    # compute phase angle (in degrees) from x-y components.
    angcnst = 1.
    angle = np.arctan2(angcnst * px, py) * 180 / np.pi
    print(np.min(angle), np.max(angle))

    # latitude string for plot title
    latstring = get_latstring(lats, latn)

    # set up plotname for saving
    plttype = "png"
    plotname = plotpath + plotname + "." + plttype

    # plot
    nlines = len(labels)
    colors = ['firebrick', 'black', 'orange', 'dodgerblue', 'seagreen']
    symbols = ['circle', 'square', 'diamond', 'x', 'triangle-up']

    scope = PlotlyScope()
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04)

    for ll in np.arange(0, nlines):
        fig.add_trace(go.Scatter(x=coh[ll, :], y=levels,
                                 mode='lines',
                                 name=labels[ll],
                                 line=dict(color=colors[ll], width=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=angle[ll, :], y=levels,
                                 mode='markers',
                                 showlegend=False,
                                 marker=dict(color=colors[ll], size=8,
                                             symbol=symbols[ll])),
                      row=1, col=2)

    fig.add_annotation(
        x=-90,
        y=50,
        xref="x2",
        yref="y2",
        text="precip lags",
        showarrow=False,
        bgcolor="white",
        opacity=0.8
    )
    fig.add_annotation(
        x=90,
        y=50,
        xref="x2",
        yref="y2",
        text="precip leads",
        showarrow=False,
        bgcolor="white",
        opacity=0.8
    )

    fig.update_layout(title=titlestr + ' ' + latstring, width=900, height=600,
                      legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01)
                      )

    fig.update_xaxes(title_text='coh^2', range=[xlim[0], xlim[1]], row=1, col=1)
    fig.update_xaxes(title_text='phase angle', range=[-180, 180], dtick=90, row=1, col=2)
    fig.update_yaxes(range=[100, 1000], dtick=100, title_text='hPa', autorange="reversed", row=1, col=1)
    fig.update_yaxes(range=[100, 1000], dtick=100, autorange="reversed", row=1, col=2)

    with open(plotname, "wb") as f:
        f.write(scope.transform(fig, format=plttype))

    return


def plot_vertcoh_panel(ds_plot, labels, titlestr, plotname, plotpath, lats, latn, xlim=[0, 0.5]):
    """
    Panel plot of averaged coherence and phase values by level.
    :param ds_plot: xarray dataset containing the data to plot. This includes nplot as an attribute and
    the source names and forecast hours. The
    :param labels: Labels for each variable, should include variable and symm/ anti-symm tag.
    :param titlestr: Title for the plot, should include CCEW/ MJO tag and variable name for var1 (usually precip).
    :param plotname: Name for the plot to be saved under.
    :param plotpath: Path for the plot to be saved at.
    :param lats: Southern latitude value the spectra were averaged over.
    :param latn: Northern latitude value the spectra were averaged over.
    :param xlim: optional parameter specifying the maximum coherence value on the x-axis
    :return:
    """

    #sources1 = ds_plot['sources1']
    #sources2 = ds_plot['sources2']
    nplot = ds_plot.attrs['nplot']
    varnames = ds_plot.data_vars

    # compute phase angle (in degrees) from x-y components.
    angcnst = 1.

    # latitude string for plot title
    latstring = get_latstring(lats, latn)

    # set up plotname for saving
    plttype = "png"
    plotname = plotpath + plotname + "." + plttype

    # plot
    nlines = len(labels)
    colors = ['firebrick', 'black', 'orange', 'dodgerblue', 'seagreen']
    symbols = ['circle', 'square', 'diamond', 'x', 'triangle-up']

    scope = PlotlyScope()
    fig = make_subplots(rows=nplot/2, cols=4, shared_yaxes=True, horizontal_spacing=0.04)

    for pp in np.arange(0, nplot/2):
        coh = ds_plot[varnames[pp*nplot]]
        coh.rename({'plev': 'levels'})
        px = ds_plot[varnames[pp*nplot+1]]
        py = ds_plot[varnames[pp * nplot + 2]]
        angle = np.arctan2(angcnst * px, py) * 180 / np.pi

        for ll in np.arange(0, nlines):
            fig.add_trace(go.Scatter(x=coh[ll, :], y=coh['levels'],
                                    mode='lines',
                                    name=labels[ll],
                                    line=dict(color=colors[ll], width=2)),
                        row=pp, col=1)
            fig.add_trace(go.Scatter(x=angle[ll, :], y=coh['levels'],
                                    mode='markers',
                                    showlegend=False,
                                    marker=dict(color=colors[ll], size=8,
                                                symbol=symbols[ll])),
                        row=pp, col=2)
        coh = ds_plot[varnames[(pp+1) * nplot/2]]
        coh.rename({'plev': 'levels'})
        px = ds_plot[varnames[(pp+1) * nplot/2 + 1]]
        py = ds_plot[varnames[(pp+1) * nplot/2 + 2]]
        angle = np.arctan2(angcnst * px, py) * 180 / np.pi

        for ll in np.arange(0, nlines):
            fig.add_trace(go.Scatter(x=coh[ll, :], y=coh['levels'],
                                     mode='lines',
                                     name=labels[ll],
                                     line=dict(color=colors[ll], width=2)),
                          row=pp, col=1)
            fig.add_trace(go.Scatter(x=angle[ll, :], y=coh['levels'],
                                     mode='markers',
                                     showlegend=False,
                                     marker=dict(color=colors[ll], size=8,
                                                 symbol=symbols[ll])),
                          row=pp, col=2)

        fig.add_annotation(
            x=-90,
            y=50,
            xref="x2",
            yref="y2",
            text="precip lags",
            showarrow=False,
            bgcolor="white",
            opacity=0.8
        )
        fig.add_annotation(
            x=90,
            y=50,
            xref="x2",
            yref="y2",
            text="precip leads",
            showarrow=False,
            bgcolor="white",
            opacity=0.8
        )

    fig.update_layout(title=titlestr + ' ' + latstring, width=900, height=600,
                      legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01)
                      )

    fig.update_xaxes(title_text='coh^2', range=[xlim[0], xlim[1]], row=1, col=1)
    fig.update_xaxes(title_text='phase angle', range=[-180, 180], dtick=90, row=1, col=2)
    fig.update_yaxes(range=[100, 1000], dtick=100, title_text='hPa', autorange="reversed", row=1, col=1)
    fig.update_yaxes(range=[100, 1000], dtick=100, autorange="reversed", row=1, col=2)

    with open(plotname, "wb") as f:
        f.write(scope.transform(fig, format=plttype))

    return

