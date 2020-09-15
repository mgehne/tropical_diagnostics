"""
Vertical coherence profile plots using plotly module. Use spacetime_plot.py to
plot full spectra
"""

import numpy as np
import plotly.graph_objects as go
from kaleido.scopes.plotly import PlotlyScope


def plot_vertcoh(coh, px, py, levels, labels, titlestr, plotname, plotpath, lats, latn):
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
    :return:
    """

    # compute phase angle (in degrees) from x-y components.
    angcnst = 1.
    angle = np.arctan2(angcnst*px, py)*180/np.pi+90

    # latitude string for plot title
    latstring = get_latstring(lats, latn)

    # set up plotname for saving
    plttype = "png"
    plotname = plotpath + plotname+"." + plttype

    # plot
    nlines = len(labels)
    scope = PlotlyScope()
    fig = go.Figure()
    for ll in np.arange(0, nlines):
        fig.add_trace(go.Scatter(x=coh[ll, :], y=levels,
                             mode='lines',
                             name=labels[ll]))

    fig.update_layout(title=titlestr+' '+latstring, yaxis=list(autorange="reversed"))

    fig.update_xaxes(ticks="", tick0=0, dtick=0.1, title_text='coh^2')
    fig.update_yaxes(ticks="", tick0=0, dtick=50, title_text='hPa')

    with open(plotname, "wb") as f:
        f.write(scope.transform(fig, format=plttype))
