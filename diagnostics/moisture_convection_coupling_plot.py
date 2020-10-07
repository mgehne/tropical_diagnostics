"""
Plotting routines and functions to assess the relationship between precipitation and column saturation fraction.
Code contributed by Brandon Wolding.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Define colormap for composites #
colors = [(10, 50, 120), (15, 75, 165), (30, 110, 200), (60, 160, 240), (80, 180, 250), (130, 210, 255),
          (160, 240, 255), (200, 250, 255), (230, 255, 255), (255, 255, 255), (255, 255, 255), (255, 250, 220),
          (255, 232, 120), (255, 192, 60), (255, 160, 0), (255, 96, 0), (255, 50, 0), (225, 20, 0), (192, 0, 0),
          (165, 0, 0)]
for list_index in range(len(colors)):
    colors[list_index] = tuple(tuple_element / 255. for tuple_element in colors[list_index])


def plot_csf_binned_precipitation_rate(csf_binned_precipitation_rate_dataset, min_number_of_obs, save_fig_boolean=False,
                                       figure_path_and_name='untitled.png'):
    """
    Plot CSF binned precipitation rate.
    :param csf_binned_precipitation_rate_dataset:
    :param min_number_of_obs: minumum number of obs per bin to include bin in plot
    :param save_fig_boolean: save figure to file True/ False
    :param figure_path_and_name: figure path and file name for saving
    :return:
    """
    bin_number_of_samples = csf_binned_precipitation_rate_dataset['bin_number_of_samples']
    bin_mean_precipitation_rate = csf_binned_precipitation_rate_dataset['bin_mean_variable']

    # Create mask for bin with insufficient obs #
    insufficient_obs_mask = bin_number_of_samples < min_number_of_obs

    # Create "centered" figure #
    fig = plt.figure(figsize=(10, 10))

    # Ask for, out of a 1x1 grid, the first axes #
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(bin_mean_precipitation_rate.BV1_bin_midpoint.where(~insufficient_obs_mask),
             bin_mean_precipitation_rate.where(~insufficient_obs_mask), color='k', linestyle='solid', linewidth=5)

    ax1.set_xlabel('Column Saturation Fraction', fontdict={'size': 24, 'weight': 'bold'})
    ax1.set_ylabel('Precipitation Rate [mm day$^{-1}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax1.set(xlim=(0, 1), ylim=(0, 75))

    # Axis 1 Ticks #
    ax1.tick_params(axis="x", direction="in", length=8, width=2, color="black")
    ax1.tick_params(axis="y", direction="in", length=8, width=2, color="black")

    ax1.tick_params(axis="x", labelsize=18, labelrotation=0, labelcolor="black")
    ax1.tick_params(axis="y", labelsize=18, labelrotation=0, labelcolor="black")

    for tick in ax1.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    for tick in ax1.yaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Percent of Total Samples', fontdict={'size': 24, 'weight': 'bold'})
    ax2.set(xlim=(0, 1), ylim=(0, 15))

    # Axis 2 Ticks #
    ax2.plot(bin_number_of_samples.BV1_bin_midpoint,
             (bin_number_of_samples / bin_number_of_samples.sum('BV1_bin_midpoint')) * 100, color='k',
             linestyle='dashed', linewidth=5)

    ax2.tick_params(axis="x", direction="in", length=8, width=2, color="black")
    ax2.tick_params(axis="y", direction="in", length=8, width=2, color="black")

    ax2.tick_params(axis="x", labelsize=18, labelrotation=0, labelcolor="black")
    ax2.tick_params(axis="y", labelsize=18, labelrotation=0, labelcolor="black")

    for tick in ax2.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    for tick in ax2.yaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    # Save figure #
    if save_fig_boolean:
        plt.savefig(figure_path_and_name, dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.0, metadata=None)


def plot_B_L_binned_precipitation_rate(binned_B_L_dataset, min_number_of_obs, save_fig_boolean=False,
                                       figure_path_and_name='untitled.png'):
    """
    Plot precipitation rate binned by B_L and distribution.
    :param binned_B_L_dataset:
    :param min_number_of_obs:
    :param save_fig_boolean:
    :param figure_path_and_name:
    :return:
    """
    bin_number_of_samples = binned_B_L_dataset['bin_number_of_samples']
    bin_mean_precipitation_rate = binned_B_L_dataset['bin_mean_precipitation_rate']

    # Create mask for regions with insufficient obs #
    insufficient_obs_mask = bin_number_of_samples < min_number_of_obs

    # Create "centered" figure #
    fig = plt.figure(figsize=(10, 10))

    # Ask for, out of a 1x1 grid, the first axes #
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(bin_mean_precipitation_rate.BV1_bin_midpoint.where(~insufficient_obs_mask),
             bin_mean_precipitation_rate.where(~insufficient_obs_mask), color='k', linestyle='solid', linewidth=5)

    ax1.set_xlabel('B$_{L}$ [m s$^{-2}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax1.set_ylabel('Precipitation Rate [mm day$^{-1}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax1.set(xlim=(-1.0, 0.5), ylim=(0, 80))

    # Axis 1 Ticks #
    ax1.tick_params(axis="x", direction="in", length=8, width=2, color="black")
    ax1.tick_params(axis="y", direction="in", length=8, width=2, color="black")

    ax1.tick_params(axis="x", labelsize=18, labelrotation=0, labelcolor="black")
    ax1.tick_params(axis="y", labelsize=18, labelrotation=0, labelcolor="black")

    for tick in ax1.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    for tick in ax1.yaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Percent of Total Samples', fontdict={'size': 24, 'weight': 'bold'})
    ax2.set(xlim=(-1.0, 0.5), ylim=(0, 15))

    # Axis 2 Ticks #
    ax2.plot(bin_number_of_samples.BV1_bin_midpoint,
             (bin_number_of_samples / bin_number_of_samples.sum('BV1_bin_midpoint')) * 100, color='k',
             linestyle='dashed', linewidth=5)

    ax2.tick_params(axis="x", direction="in", length=8, width=2, color="black")
    ax2.tick_params(axis="y", direction="in", length=8, width=2, color="black")

    ax2.tick_params(axis="x", labelsize=18, labelrotation=0, labelcolor="black")
    ax2.tick_params(axis="y", labelsize=18, labelrotation=0, labelcolor="black")

    for tick in ax2.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    for tick in ax2.yaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    # Save figure #
    if save_fig_boolean:
        plt.savefig(figure_path_and_name, dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.0, metadata=None)


def plot_CSF_precipitation_rate_composites(binned_csf_precipitation_rate_dataset, min_number_of_obs,
                                           save_fig_boolean=False, figure_path_and_name='untitled.png'):
    """
    Plot composite evolution of precipitation and CSF.
    :param binned_csf_precipitation_rate_dataset: binned data set
    :param min_number_of_obs: minimum number of obs in bin for plotting
    :param save_fig_boolean: save figure True/ False
    :param figure_path_and_name: path and filename to save figure to
    :return:
    """
    bin_number_of_samples_centered = binned_csf_precipitation_rate_dataset['bin_number_of_samples_centered']
    bin_mean_delta_csf_centered = binned_csf_precipitation_rate_dataset['bin_mean_delta_csf_centered']
    bin_mean_delta_precipitation_rate_centered = binned_csf_precipitation_rate_dataset[
        'bin_mean_delta_precipitation_rate_centered']
    bin_number_pos_delta_csf_centered = binned_csf_precipitation_rate_dataset['bin_number_pos_delta_csf_centered']
    bin_number_pos_delta_precipitation_rate_centered = binned_csf_precipitation_rate_dataset[
        'bin_number_pos_delta_precipitation_rate_centered']

    # Create mask for regions with insufficient obs #
    insufficient_obs_mask = bin_number_of_samples_centered < min_number_of_obs

    n_bin = 200
    cmap_name = 'colors'
    colormap_colors = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

    # Create "centered" figure #
    fig = plt.figure(figsize=(10, 10))

    # Ask for, out of a 1x1 grid, the first axes #
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Column Saturation Fraction', fontdict={'size': 24, 'weight': 'bold'})
    ax.set_ylabel('Precipitation Rate [mm day$^{-1}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax.set(xlim=(0.3, bin_number_of_samples_centered.BV1_bin_midpoint.max()),
           ylim=(bin_number_of_samples_centered.BV2_bin_midpoint.min(), 75))

    # Axis Ticks #
    ax.tick_params(axis="x", direction="in", length=12, width=3, color="black")
    ax.tick_params(axis="y", direction="in", length=12, width=3, color="black")

    ax.tick_params(axis="x", labelsize=18, labelrotation=0, labelcolor="black")
    ax.tick_params(axis="y", labelsize=18, labelrotation=90, labelcolor="black")

    for tick in ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    # Create "meshgrid" for contour plotting #
    CSF_bin_midpoint_meshgrid, precip_bin_midpoint_meshgrid = np.meshgrid(
        bin_number_of_samples_centered.BV1_bin_midpoint, bin_number_of_samples_centered.BV2_bin_midpoint)

    CSF_bin_midpoint_meshgrid_DA = bin_number_of_samples_centered.copy()
    CSF_bin_midpoint_meshgrid_DA.values = CSF_bin_midpoint_meshgrid

    precip_bin_midpoint_meshgrid_DA = bin_number_of_samples_centered.copy()
    precip_bin_midpoint_meshgrid_DA.values = precip_bin_midpoint_meshgrid

    # make contour plot
    c = ax.contourf(CSF_bin_midpoint_meshgrid_DA, precip_bin_midpoint_meshgrid_DA,
                    (bin_number_pos_delta_csf_centered.where(~insufficient_obs_mask) / bin_number_of_samples_centered),
                    levels=np.arange(0, 1.01, .01), cmap=colormap_colors, vmin=0.0, vmax=1.0)

    # Speckle regions with insufficient observations #
    ax.plot(CSF_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask),
            precip_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask), 'ko', ms=1);

    # Quiver the bin mean leading tendency # Use this if you want to reduce number of quivers plotted
    # y_quiver_plotting_indices = list(np.arange(0,12,2)) +
    # list(np.arange(11, len(bin_number_of_samples_centered.BV2_bin_midpoint), 1))
    # q = ax.quiver(CSF_bin_midpoint_meshgrid_DA[y_quiver_plotting_indices, ::2],
    #              precip_bin_midpoint_meshgrid_DA[y_quiver_plotting_indices, ::2],\
    #              bin_mean_delta_csf_centered.where(~insufficient_obs_mask)[y_quiver_plotting_indices, ::2],
    #              bin_mean_delta_precipitation_rate_centered.where(~insufficient_obs_mask)
    #              [y_quiver_plotting_indices, ::2], width=0.007, angles='xy', scale_units='xy', scale=1, pivot='mid')
    # Very important to have "angles" and "scale_units" set to "xy". "pivot=mid" shifts so arrow center at bin center.
    # other options are "tail" and "tip"

    # Quiver the bin mean leading tendency # Use this to plot all quivers
    q = ax.quiver(CSF_bin_midpoint_meshgrid_DA, precip_bin_midpoint_meshgrid_DA,
                  bin_mean_delta_csf_centered.where(~insufficient_obs_mask),
                  bin_mean_delta_precipitation_rate_centered.where(~insufficient_obs_mask), width=0.007, angles='xy',
                  scale_units='xy', scale=1,
                  pivot='mid')
    # Very important to have "angles" and "scale_units" set to "xy". "pivot=mid" shifts so arrow center at bin center.
    # other options are "tail" and "tip"
    # ax.quiverkey(q, X=0, Y=0, U=10, label='Quiver key, length = 1', labelpos='E')

    # Colorbar #
    cbar = fig.colorbar(c, ax=ax, orientation="horizontal", pad=0.10)
    cbar.set_ticks(np.arange(0, 1.1, 0.1))
    cbar.ax.get_yaxis().labelpad = 0
    cbar.set_label('[positive tendency fraction]', rotation=0, fontdict={'size': 18, 'weight': 'bold'})
    for tick in cbar.ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(length=10, direction='in')

    # Save figure #
    if save_fig_boolean:
        plt.savefig(figure_path_and_name, dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.0, metadata=None)


def plot_CSF_precipitation_rate_difference_composites(binned_csf_precipitation_rate_dataset_baseline,
                                                      binned_csf_precipitation_rate_dataset_other, min_number_of_obs,
                                                      save_fig_boolean=False, figure_path_and_name='untitled.png'):
    """
    Plot the difference in precipitation - CSF evolution between 2 data sets. Could be the difference between obs and
    a model, or two models for example.
    :param binned_csf_precipitation_rate_dataset_baseline: first binned data set
    :param binned_csf_precipitation_rate_dataset_other: second binned data set
    :param min_number_of_obs: minimum number of obs per bin for plotting
    :param save_fig_boolean: save figure to file True/ False
    :param figure_path_and_name: figure path and filename for saving
    :return:
    """
    # Baseline dataset
    bin_number_of_samples_centered_baseline = binned_csf_precipitation_rate_dataset_baseline[
        'bin_number_of_samples_centered']
    bin_mean_delta_csf_centered_baseline = binned_csf_precipitation_rate_dataset_baseline['bin_mean_delta_csf_centered']
    bin_mean_delta_precipitation_rate_centered_baseline = binned_csf_precipitation_rate_dataset_baseline[
        'bin_mean_delta_precipitation_rate_centered']
    bin_number_pos_delta_csf_centered_baseline = binned_csf_precipitation_rate_dataset_baseline[
        'bin_number_pos_delta_csf_centered']
    bin_number_pos_delta_precipitation_rate_centered_baseline = binned_csf_precipitation_rate_dataset_baseline[
        'bin_number_pos_delta_precipitation_rate_centered']

    # Other dataset
    bin_number_of_samples_centered_other = binned_csf_precipitation_rate_dataset_other['bin_number_of_samples_centered']
    bin_mean_delta_csf_centered_other = binned_csf_precipitation_rate_dataset_other['bin_mean_delta_csf_centered']
    bin_mean_delta_precipitation_rate_centered_other = binned_csf_precipitation_rate_dataset_other[
        'bin_mean_delta_precipitation_rate_centered']
    bin_number_pos_delta_csf_centered_other = binned_csf_precipitation_rate_dataset_other[
        'bin_number_pos_delta_csf_centered']
    bin_number_pos_delta_precipitation_rate_centered_other = binned_csf_precipitation_rate_dataset_other[
        'bin_number_pos_delta_precipitation_rate_centered']

    # Calculate the difference between the "other" dataset and the "baseline" dataset
    bin_mean_delta_csf_centered_difference = bin_mean_delta_csf_centered_other - bin_mean_delta_csf_centered_baseline
    bin_mean_delta_precipitation_rate_centered_difference = \
        bin_mean_delta_precipitation_rate_centered_other - bin_mean_delta_precipitation_rate_centered_baseline

    # Create mask for regions with insufficient obs #
    insufficient_obs_mask = np.logical_or((bin_number_of_samples_centered_baseline.values < min_number_of_obs),
                                          (bin_number_of_samples_centered_other.values < min_number_of_obs))

    n_bin = 200
    cmap_name = 'colors'
    colormap_colors = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

    # Create "centered" figure #
    fig = plt.figure(figsize=(10, 10))

    # Ask for, out of a 1x1 grid, the first axes #
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Column Saturation Fraction', fontdict={'size': 24, 'weight': 'bold'})
    ax.set_ylabel('Precipitation Rate [mm day$^{-1}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax.set(xlim=(0.3, bin_number_of_samples_centered_other.BV1_bin_midpoint.max()),
           ylim=(bin_number_of_samples_centered_other.BV2_bin_midpoint.min(), 75))

    # Axis Ticks #
    ax.tick_params(axis="x", direction="in", length=12, width=3, color="black")
    ax.tick_params(axis="y", direction="in", length=12, width=3, color="black")

    ax.tick_params(axis="x", labelsize=18, labelrotation=0, labelcolor="black")
    ax.tick_params(axis="y", labelsize=18, labelrotation=90, labelcolor="black")

    for tick in ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    # Create "meshgrid" for contour plotting #
    CSF_bin_midpoint_meshgrid, precip_bin_midpoint_meshgrid = np.meshgrid(
        bin_number_of_samples_centered_other.BV1_bin_midpoint, bin_number_of_samples_centered_other.BV2_bin_midpoint)

    CSF_bin_midpoint_meshgrid_DA = bin_number_of_samples_centered_other.copy()
    CSF_bin_midpoint_meshgrid_DA.values = CSF_bin_midpoint_meshgrid

    precip_bin_midpoint_meshgrid_DA = bin_number_of_samples_centered_other.copy()
    precip_bin_midpoint_meshgrid_DA.values = precip_bin_midpoint_meshgrid

    # Contourf #
    c = ax.contourf(CSF_bin_midpoint_meshgrid_DA, precip_bin_midpoint_meshgrid_DA, (
                bin_number_pos_delta_csf_centered_other.where(
                    ~insufficient_obs_mask) / bin_number_of_samples_centered_other) - (
                                bin_number_pos_delta_csf_centered_baseline.where(
                                    ~insufficient_obs_mask) / bin_number_of_samples_centered_baseline),
                    levels=np.arange(-0.1, 0.11, .01), cmap=colormap_colors, vmin=-0.1, vmax=0.1, extend='both')

    # Speckle regions with insufficient observations #
    ax.plot(CSF_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask),
            precip_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask), 'ko', ms=1);

    # Quiver the bin mean leading tendency # Use this if you want to reduce number of quivers plotted
    #y_quiver_plotting_indices = list(np.arange(0,12,2)) + \
    #                            list(np.arange(11, len(bin_number_of_samples_centered.BV2_bin_midpoint), 1))
    #q = ax.quiver(CSF_bin_midpoint_meshgrid_DA[y_quiver_plotting_indices, ::2],
    #              precip_bin_midpoint_meshgrid_DA[y_quiver_plotting_indices, ::2],\
    #              bin_mean_delta_csf_centered_difference.where(~insufficient_obs_mask)[y_quiver_plotting_indices, ::2],
    #              bin_mean_delta_precipitation_rate_centered_difference.where(~insufficient_obs_mask)
    #              [y_quiver_plotting_indices, ::2], width=0.007, angles='xy', scale_units='xy', scale=1, pivot='mid')
    # Very important to have "angles" and "scale_units" set to "xy". "pivot=mid" shifts so arrow center at bin center.
    # other options are "tail" and "tip"

    # Quiver the bin mean leading tendency # Use this to plot all quivers
    q = ax.quiver(CSF_bin_midpoint_meshgrid_DA, precip_bin_midpoint_meshgrid_DA,
                  bin_mean_delta_csf_centered_difference.where(~insufficient_obs_mask),
                  bin_mean_delta_precipitation_rate_centered_difference.where(~insufficient_obs_mask), width=0.007,
                  angles='xy', scale_units='xy', scale=1,
                  pivot='mid')
    # Very important to have "angles" and "scale_units" set to "xy". "pivot=mid" shifts so arrow center at bin center.
    # other options are "tail" and "tip"

    # ax.quiverkey(q, X=0, Y=0, U=10, label='Quiver key, length = 1', labelpos='E')

    # Colorbar #
    cbar = fig.colorbar(c, ax=ax, orientation="horizontal", pad=0.10)
    cbar.set_ticks(np.arange(-0.1, 0.15, 0.05))
    cbar.ax.get_yaxis().labelpad = 0
    cbar.set_label('[positive tendency fraction]', rotation=0, fontdict={'size': 18, 'weight': 'bold'})
    for tick in cbar.ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(length=10, direction='in')

    # Save figure #
    if save_fig_boolean:
        plt.savefig(figure_path_and_name, dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.0, metadata=None)


def plot_undilute_B_L_VS_dilution_composites_V1(binned_undilute_B_L_dilution_dataset, min_number_of_obs,
                                                save_fig_boolean=False, figure_path_and_name='untitled.png'):
    """
    Plot composites of undilute B_L against dilution and vectors. Only plot bins with at least min_number_of_obs.
    :param binned_undilute_B_L_dilution_dataset: data set containing B_L components
    :param min_number_of_obs: minimum number of obs per bin to plot
    :param save_fig_boolean: save figure or not
    :param figure_path_and_name: filename to save the figure to
    :return:
    """
    # Plotting scale factors
    arrow_scale_factor = 3.

    # Load data
    bin_number_of_samples = binned_undilute_B_L_dilution_dataset['bin_number_of_samples']
    bin_mean_precipitation_rate = binned_undilute_B_L_dilution_dataset['bin_mean_precipitation_rate']

    bin_mean_delta_dilution_leading = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_leading']
    bin_mean_delta_undilute_B_L_leading = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_leading']
    bin_number_pos_delta_dilution_leading = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_dilution_leading']
    bin_number_pos_delta_undilute_B_L_leading = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_undilute_B_L_leading']

    bin_mean_delta_dilution_lagging = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_lagging']
    bin_mean_delta_undilute_B_L_lagging = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_lagging']
    bin_number_pos_delta_dilution_lagging = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_dilution_lagging']
    bin_number_pos_delta_undilute_B_L_lagging = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_undilute_B_L_lagging']

    bin_mean_delta_dilution_centered = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_centered']
    bin_mean_delta_undilute_B_L_centered = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_centered']
    bin_number_pos_delta_dilution_centered = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_dilution_centered']
    bin_number_pos_delta_undilute_B_L_centered = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_undilute_B_L_centered']

    # Create mask for regions with insufficient obs #
    insufficient_obs_mask = bin_number_of_samples < min_number_of_obs

    n_bin = 200
    cmap_name = 'colors'
    colormap_colors = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

    # Create "centered" figure #
    fig = plt.figure(figsize=(10, 10))

    # Ask for, out of a 1x1 grid, the first axes #
    x_axis_limit = (-0.2, 0)
    x_tick_spacing = 0.04

    y_axis_limit = (-0.1, 0.1)
    y_tick_spacing = 0.04

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Dilution of B$_{L}$ [m s$^{-2}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax.set_ylabel('Undilute B$_{L}$  [m s$^{-2}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax.set(xlim=(x_axis_limit[0], x_axis_limit[1]), ylim=(y_axis_limit[0], y_axis_limit[1]))

    # Axis Ticks #
    plt.xticks(np.arange(x_axis_limit[0], x_axis_limit[1] + x_tick_spacing, x_tick_spacing))
    plt.yticks(np.arange(y_axis_limit[0], y_axis_limit[1] + y_tick_spacing, y_tick_spacing))

    ax.tick_params(axis="x", direction="in", length=12, width=3, color="black")
    ax.tick_params(axis="y", direction="in", length=12, width=3, color="black")

    ax.tick_params(axis="x", labelsize=18, labelrotation=0, labelcolor="black")
    ax.tick_params(axis="y", labelsize=18, labelrotation=0, labelcolor="black")

    for tick in ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Create "meshgrid" for contour plotting #
    BV1_bin_midpoint_meshgrid, BV2_bin_midpoint_meshgrid = np.meshgrid(bin_number_of_samples.BV1_bin_midpoint,
                                                                       bin_number_of_samples.BV2_bin_midpoint)

    BV1_bin_midpoint_meshgrid_DA = bin_number_of_samples.copy()
    BV1_bin_midpoint_meshgrid_DA.values = BV1_bin_midpoint_meshgrid

    BV2_bin_midpoint_meshgrid_DA = bin_number_of_samples.copy()
    BV2_bin_midpoint_meshgrid_DA.values = BV2_bin_midpoint_meshgrid

    # Contourf bin mean precipitation rate#
    c = ax.contourf(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    bin_mean_precipitation_rate.where(~insufficient_obs_mask), levels=np.arange(0, 21, 1),
                    cmap=colormap_colors, vmin=0.0, vmax=20.0, extend='max')

    # Contour bin mean B_L #
    c2 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=np.arange(-1, 0, 0.05), colors='k', linestyles='dashed', linewidths=3)

    c3 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=0, colors='k', linestyles='solid', linewidths=6)

    c4 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=np.arange(0.1, 2.1, 0.05), colors='k', linestyles='solid', linewidths=3)

    # Speckle regions with insufficient observations #
    ax.plot(BV1_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask),
            BV2_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask), 'ko', ms=1);

    # Mark the mode of observations #
    ax.plot(BV1_bin_midpoint_meshgrid_DA.where(bin_number_of_samples == bin_number_of_samples.max()),
            BV2_bin_midpoint_meshgrid_DA.where(bin_number_of_samples == bin_number_of_samples.max()), 'ks', ms=20,
            color=[0.7, 0.7, 0.7], fillstyle='full', markeredgecolor='k', markeredgewidth=3);

    # Quiver the bin mean centered tendency #
    # Very important to have "angles" and "scale_units" set to "xy".
    # "pivot=mid" shifts so arrow center at bin center. other options are "tail" and "tip"
    q = ax.quiver(BV1_bin_midpoint_meshgrid_DA[::2, ::2], BV2_bin_midpoint_meshgrid_DA[::2, ::2],
                  bin_mean_delta_dilution_centered.where(~insufficient_obs_mask)[::2, ::2],
                  bin_mean_delta_undilute_B_L_centered.where(~insufficient_obs_mask)[::2, ::2], width=0.007,
                  angles='xy', scale_units='xy', scale=(1 / arrow_scale_factor),
                  pivot='mid')

    # Colorbar #
    cbar = fig.colorbar(c, ax=ax, orientation="horizontal", pad=0.14)
    cbar.set_ticks(np.arange(0, 22, 2))
    cbar.ax.get_yaxis().labelpad = 0
    cbar.set_label('Precipitation Rate [mm day$^{-1}$]', rotation=0, fontdict={'size': 18, 'weight': 'bold'})
    for tick in cbar.ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(length=10, direction='in')

    # Save figure #
    if save_fig_boolean:
        plt.savefig(figure_path_and_name, dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.0, metadata=None)


def plot_undilute_B_L_VS_dilution_composites_log_precipitation(binned_undilute_B_L_dilution_dataset, min_number_of_obs,
                                                               save_fig_boolean=False,
                                                               figure_path_and_name='untitled.png'):
    """
    Plot composites of undilute B_L against dilution and vectors using log precipitation. Only plot bins with at
    least min_number_of_obs.
    :param binned_undilute_B_L_dilution_dataset: data set containing B_L components
    :param min_number_of_obs: minimum number of obs per bin to plot
    :param save_fig_boolean: save figure or not
    :param figure_path_and_name: filename to save the figure to
    :return:
    """
    # Plotting scale factors
    arrow_scale_factor = 3.

    # Load data
    bin_number_of_samples = binned_undilute_B_L_dilution_dataset['bin_number_of_samples']
    bin_mean_precipitation_rate = binned_undilute_B_L_dilution_dataset['bin_mean_precipitation_rate']

    bin_mean_delta_dilution_leading = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_leading']
    bin_mean_delta_undilute_B_L_leading = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_leading']
    bin_number_pos_delta_dilution_leading = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_dilution_leading']
    bin_number_pos_delta_undilute_B_L_leading = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_undilute_B_L_leading']

    bin_mean_delta_dilution_lagging = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_lagging']
    bin_mean_delta_undilute_B_L_lagging = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_lagging']
    bin_number_pos_delta_dilution_lagging = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_dilution_lagging']
    bin_number_pos_delta_undilute_B_L_lagging = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_undilute_B_L_lagging']

    bin_mean_delta_dilution_centered = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_centered']
    bin_mean_delta_undilute_B_L_centered = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_centered']
    bin_number_pos_delta_dilution_centered = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_dilution_centered']
    bin_number_pos_delta_undilute_B_L_centered = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_undilute_B_L_centered']

    # Create mask for regions with insufficient obs #
    insufficient_obs_mask = bin_number_of_samples < min_number_of_obs

    n_bin = 200
    cmap_name = 'colors'
    colormap_colors = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

    # Create "centered" figure #
    fig = plt.figure(figsize=(10, 10))

    # Ask for, out of a 1x1 grid, the first axes #
    x_axis_limit = (-0.2, 0)
    x_tick_spacing = 0.04
    y_axis_limit = (-0.1, 0.1)
    y_tick_spacing = 0.04

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Dilution of B$_{L}$ [m s$^{-2}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax.set_ylabel('Undilute B$_{L}$  [m s$^{-2}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax.set(xlim=(x_axis_limit[0], x_axis_limit[1]), ylim=(y_axis_limit[0], y_axis_limit[1]))

    # Axis Ticks #
    plt.xticks(np.arange(x_axis_limit[0], x_axis_limit[1] + x_tick_spacing, x_tick_spacing))
    plt.yticks(np.arange(y_axis_limit[0], y_axis_limit[1] + y_tick_spacing, y_tick_spacing))

    ax.tick_params(axis="x", direction="in", length=12, width=3, color="black")
    ax.tick_params(axis="y", direction="in", length=12, width=3, color="black")

    ax.tick_params(axis="x", labelsize=18, labelrotation=0, labelcolor="black")
    ax.tick_params(axis="y", labelsize=18, labelrotation=0, labelcolor="black")

    for tick in ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    # Create "meshgrid" for contour plotting #
    BV1_bin_midpoint_meshgrid, BV2_bin_midpoint_meshgrid = np.meshgrid(bin_number_of_samples.BV1_bin_midpoint,
                                                                       bin_number_of_samples.BV2_bin_midpoint)

    BV1_bin_midpoint_meshgrid_DA = bin_number_of_samples.copy()
    BV1_bin_midpoint_meshgrid_DA.values = BV1_bin_midpoint_meshgrid

    BV2_bin_midpoint_meshgrid_DA = bin_number_of_samples.copy()
    BV2_bin_midpoint_meshgrid_DA.values = BV2_bin_midpoint_meshgrid

    # Contourf bin mean precipitation rate#
    c = ax.contourf(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    np.log10(bin_mean_precipitation_rate.where(~insufficient_obs_mask)), levels=np.arange(0, 2.1, 0.1),
                    cmap=colormap_colors, vmin=0, vmax=2.0, extend='both')

    # Contour bin mean B_L #
    c2 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=np.arange(-1, 0, 0.05), colors='k', linestyles='dashed', linewidths=3)

    c3 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=0, colors='k', linestyles='solid', linewidths=6)

    c4 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=np.arange(0.1, 2.1, 0.05), colors='k', linestyles='solid', linewidths=3)

    # Speckle regions with insufficient observations #
    ax.plot(BV1_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask),
            BV2_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask), 'ko', ms=1);

    # Mark the mode of observations #
    ax.plot(BV1_bin_midpoint_meshgrid_DA.where(bin_number_of_samples == bin_number_of_samples.max()),
            BV2_bin_midpoint_meshgrid_DA.where(bin_number_of_samples == bin_number_of_samples.max()), 'ks', ms=20,
            color=[0.7, 0.7, 0.7], fillstyle='full', markeredgecolor='k', markeredgewidth=3)

    # Quiver the bin mean centered tendency #
    # Very important to have "angles" and "scale_units" set to "xy".
    # "pivot=mid" shifts so arrow center at bin center. other options are "tail" and "tip"
    q = ax.quiver(BV1_bin_midpoint_meshgrid_DA[::2, ::2], BV2_bin_midpoint_meshgrid_DA[::2, ::2],
                  bin_mean_delta_dilution_centered.where(~insufficient_obs_mask)[::2, ::2],
                  bin_mean_delta_undilute_B_L_centered.where(~insufficient_obs_mask)[::2, ::2], width=0.007,
                  angles='xy', scale_units='xy', scale=(1 / arrow_scale_factor),
                  pivot='mid')

    # Colorbar #
    cbar = fig.colorbar(c, ax=ax, orientation="horizontal", pad=0.14)
    cbar.set_ticks(np.arange(0, 2.2, 0.2))
    cbar.ax.get_yaxis().labelpad = 0
    cbar.set_label('Log$_{10}$ Precipitation Rate [mm day$^{-1}$]', rotation=0, fontdict={'size': 18, 'weight': 'bold'})
    for tick in cbar.ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(length=10, direction='in')

    # Save figure #
    if save_fig_boolean:
        plt.savefig(figure_path_and_name, dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.0, metadata=None)


def plot_undilute_B_L_VS_dilution_composites_V1_zoom_out(binned_undilute_B_L_dilution_dataset, min_number_of_obs,
                                                         save_fig_boolean=False, figure_path_and_name='untitled.png'):
    """
    Plot composites of undilute B_L against dilution and vectors. Only plot bins with at least min_number_of_obs.
    This is the zoomed out version of the plot showing a larger region in undilute B_L - dilution space.
    :param binned_undilute_B_L_dilution_dataset: data set containing B_L components
    :param min_number_of_obs: minimum number of obs per bin to plot
    :param save_fig_boolean: save figure or not
    :param figure_path_and_name: filename to save the figure to
    :return:
    """
    # Plotting scale factors
    arrow_scale_factor = 3.

    # Load data
    bin_number_of_samples = binned_undilute_B_L_dilution_dataset['bin_number_of_samples']
    bin_mean_precipitation_rate = binned_undilute_B_L_dilution_dataset['bin_mean_precipitation_rate']

    bin_mean_delta_dilution_leading = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_leading']
    bin_mean_delta_undilute_B_L_leading = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_leading']
    bin_number_pos_delta_dilution_leading = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_dilution_leading']
    bin_number_pos_delta_undilute_B_L_leading = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_undilute_B_L_leading']

    bin_mean_delta_dilution_lagging = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_lagging']
    bin_mean_delta_undilute_B_L_lagging = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_lagging']
    bin_number_pos_delta_dilution_lagging = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_dilution_lagging']
    bin_number_pos_delta_undilute_B_L_lagging = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_undilute_B_L_lagging']

    bin_mean_delta_dilution_centered = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_centered']
    bin_mean_delta_undilute_B_L_centered = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_centered']
    bin_number_pos_delta_dilution_centered = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_dilution_centered']
    bin_number_pos_delta_undilute_B_L_centered = \
        binned_undilute_B_L_dilution_dataset['bin_number_pos_delta_undilute_B_L_centered']

    # Create mask for regions with insufficient obs #
    insufficient_obs_mask = bin_number_of_samples < min_number_of_obs

    n_bin = 200
    cmap_name = 'colors'
    colormap_colors = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

    # Create "centered" figure #
    fig = plt.figure(figsize=(10, 10))

    # Ask for, out of a 1x1 grid, the first axes #
    x_axis_limit = (-0.4, 0)
    x_tick_spacing = 0.08
    y_axis_limit = (-0.24, 0.1)
    y_tick_spacing = 0.08

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Dilution of B$_{L}$ [m s$^{-2}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax.set_ylabel('Undilute B$_{L}$  [m s$^{-2}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax.set(xlim=(x_axis_limit[0], x_axis_limit[1]), ylim=(y_axis_limit[0], y_axis_limit[1]))

    # Axis Ticks #
    plt.xticks(np.arange(x_axis_limit[0], x_axis_limit[1] + x_tick_spacing, x_tick_spacing))
    plt.yticks(np.arange(y_axis_limit[0], y_axis_limit[1] + y_tick_spacing, y_tick_spacing))

    ax.tick_params(axis="x", direction="in", length=12, width=3, color="black")
    ax.tick_params(axis="y", direction="in", length=12, width=3, color="black")

    ax.tick_params(axis="x", labelsize=18, labelrotation=0, labelcolor="black")
    ax.tick_params(axis="y", labelsize=18, labelrotation=0, labelcolor="black")

    for tick in ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    # Create "meshgrid" for contour plotting #
    BV1_bin_midpoint_meshgrid, BV2_bin_midpoint_meshgrid = np.meshgrid(bin_number_of_samples.BV1_bin_midpoint,
                                                                       bin_number_of_samples.BV2_bin_midpoint)

    BV1_bin_midpoint_meshgrid_DA = bin_number_of_samples.copy()
    BV1_bin_midpoint_meshgrid_DA.values = BV1_bin_midpoint_meshgrid

    BV2_bin_midpoint_meshgrid_DA = bin_number_of_samples.copy()
    BV2_bin_midpoint_meshgrid_DA.values = BV2_bin_midpoint_meshgrid

    # Contourf bin mean precipitation rate #
    c = ax.contourf(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    bin_mean_precipitation_rate.where(~insufficient_obs_mask), levels=np.arange(0, 21, 1),
                    cmap=colormap_colors, vmin=0.0, vmax=20.0, extend='max')

    # Contour bin mean B_L #

    c2 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=np.arange(-1, 0, 0.05), colors='k', linestyles='dashed', linewidths=3)

    c3 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=0, colors='k', linestyles='solid', linewidths=6)

    c4 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=np.arange(0.1, 2.1, 0.05), colors='k', linestyles='solid', linewidths=3)

    # Speckle regions with insufficient observations #
    ax.plot(BV1_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask),
            BV2_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask), 'ko', ms=1);

    # Mark the mode of observations #
    ax.plot(BV1_bin_midpoint_meshgrid_DA.where(bin_number_of_samples == bin_number_of_samples.max()),
            BV2_bin_midpoint_meshgrid_DA.where(bin_number_of_samples == bin_number_of_samples.max()), 'ks', ms=20,
            color=[0.7, 0.7, 0.7], fillstyle='full', markeredgecolor='k', markeredgewidth=3);

    # Quiver the bin mean centered tendency #
    # Very important to have "angles" and "scale_units" set to "xy".
    # "pivot=mid" shifts so arrow center at bin center. other options are "tail" and "tip"
    q = ax.quiver(BV1_bin_midpoint_meshgrid_DA[::2, ::2], BV2_bin_midpoint_meshgrid_DA[::2, ::2],
                  bin_mean_delta_dilution_centered.where(~insufficient_obs_mask)[::2, ::2],
                  bin_mean_delta_undilute_B_L_centered.where(~insufficient_obs_mask)[::2, ::2], width=0.007,
                  angles='xy', scale_units='xy', scale=(1 / arrow_scale_factor),
                  pivot='mid')

    # Colorbar #
    cbar = fig.colorbar(c, ax=ax, orientation="horizontal", pad=0.14)
    cbar.set_ticks(np.arange(0, 22, 2))
    cbar.ax.get_yaxis().labelpad = 0
    cbar.set_label('Precipitation Rate [mm day$^{-1}$]', rotation=0, fontdict={'size': 18, 'weight': 'bold'})
    for tick in cbar.ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(length=10, direction='in')

    # Save figure #
    if save_fig_boolean:
        plt.savefig(figure_path_and_name, dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.0, metadata=None)


def plot_undilute_B_L_VS_dilution_composites_log_precipitation_zoom_out(binned_undilute_B_L_dilution_dataset,
                                                                        min_number_of_obs, save_fig_boolean=False,
                                                                        figure_path_and_name='untitled.png'):
    """
    Zoomed out log precipitation undilute B_L vs dilution composites plot with vectors. Only plot bins with
    a minimum number of observations in them.
    :param binned_undilute_B_L_dilution_dataset:
    :param min_number_of_obs:
    :param save_fig_boolean:
    :param figure_path_and_name:
    :return:
    """
    # Plotting scale factors
    arrow_scale_factor = 3.

    # Load data
    bin_number_of_samples = binned_undilute_B_L_dilution_dataset['bin_number_of_samples']
    bin_mean_precipitation_rate = binned_undilute_B_L_dilution_dataset['bin_mean_precipitation_rate']

    bin_mean_delta_dilution_leading = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_leading']
    bin_mean_delta_undilute_B_L_leading = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_leading']
    bin_number_pos_delta_dilution_leading = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_dilution_leading']
    bin_number_pos_delta_undilute_B_L_leading = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_undilute_B_L_leading']

    bin_mean_delta_dilution_lagging = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_lagging']
    bin_mean_delta_undilute_B_L_lagging = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_lagging']
    bin_number_pos_delta_dilution_lagging = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_dilution_lagging']
    bin_number_pos_delta_undilute_B_L_lagging = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_undilute_B_L_lagging']

    bin_mean_delta_dilution_centered = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_centered']
    bin_mean_delta_undilute_B_L_centered = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_centered']
    bin_number_pos_delta_dilution_centered = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_dilution_centered']
    bin_number_pos_delta_undilute_B_L_centered = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_undilute_B_L_centered']

    # Create mask for regions with insufficient obs #
    insufficient_obs_mask = bin_number_of_samples < min_number_of_obs

    n_bin = 200
    cmap_name = 'colors'
    colormap_colors = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

    # Create "centered" figure #
    fig = plt.figure(figsize=(10, 10))

    # Ask for, out of a 1x1 grid, the first axes #
    x_axis_limit = (-0.4, 0)
    x_tick_spacing = 0.08
    y_axis_limit = (-0.24, 0.1)
    y_tick_spacing = 0.08

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Dilution of B$_{L}$ [m s$^{-2}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax.set_ylabel('Undilute B$_{L}$  [m s$^{-2}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax.set(xlim=(x_axis_limit[0], x_axis_limit[1]), ylim=(y_axis_limit[0], y_axis_limit[1]))

    # Axis Ticks #
    plt.xticks(np.arange(x_axis_limit[0], x_axis_limit[1] + x_tick_spacing, x_tick_spacing))
    plt.yticks(np.arange(y_axis_limit[0], y_axis_limit[1] + y_tick_spacing, y_tick_spacing))

    ax.tick_params(axis="x", direction="in", length=12, width=3, color="black")
    ax.tick_params(axis="y", direction="in", length=12, width=3, color="black")

    ax.tick_params(axis="x", labelsize=18, labelrotation=0, labelcolor="black")
    ax.tick_params(axis="y", labelsize=18, labelrotation=0, labelcolor="black")

    for tick in ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    # Create "meshgrid" for contour plotting #
    BV1_bin_midpoint_meshgrid, BV2_bin_midpoint_meshgrid = np.meshgrid(bin_number_of_samples.BV1_bin_midpoint,
                                                                       bin_number_of_samples.BV2_bin_midpoint)

    BV1_bin_midpoint_meshgrid_DA = bin_number_of_samples.copy()
    BV1_bin_midpoint_meshgrid_DA.values = BV1_bin_midpoint_meshgrid

    BV2_bin_midpoint_meshgrid_DA = bin_number_of_samples.copy()
    BV2_bin_midpoint_meshgrid_DA.values = BV2_bin_midpoint_meshgrid

    # Contourf bin mean precipitation rate#
    c = ax.contourf(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    np.log10(bin_mean_precipitation_rate.where(~insufficient_obs_mask)), levels=np.arange(0, 2.1, 0.1),
                    cmap=colormap_colors, vmin=0, vmax=2.0, extend='both')

    # Contour bin mean B_L #
    c2 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=np.arange(-1, 0, 0.05), colors='k', linestyles='dashed', linewidths=3)

    c3 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=0, colors='k', linestyles='solid', linewidths=6)

    c4 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=np.arange(0.1, 2.1, 0.05), colors='k', linestyles='solid', linewidths=3)

    # Speckle regions with insufficient observations #
    ax.plot(BV1_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask),
            BV2_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask), 'ko', ms=1)

    # Mark the mode of observations #
    ax.plot(BV1_bin_midpoint_meshgrid_DA.where(bin_number_of_samples == bin_number_of_samples.max()),
            BV2_bin_midpoint_meshgrid_DA.where(bin_number_of_samples == bin_number_of_samples.max()), 'ks', ms=20,
            color=[0.7, 0.7, 0.7], fillstyle='full', markeredgecolor='k', markeredgewidth=3)

    # Quiver the bin mean centered tendency #
    # Very important to have "angles" and "scale_units" set to "xy".
    # "pivot=mid" shifts so arrow center at bin center. other options are "tail" and "tip"
    q = ax.quiver(BV1_bin_midpoint_meshgrid_DA[::2, ::2], BV2_bin_midpoint_meshgrid_DA[::2, ::2],
                  bin_mean_delta_dilution_centered.where(~insufficient_obs_mask)[::2, ::2],
                  bin_mean_delta_undilute_B_L_centered.where(~insufficient_obs_mask)[::2, ::2], width=0.007,
                  angles='xy', scale_units='xy', scale=(1 / arrow_scale_factor),  pivot = 'mid')

    # Colorbar #
    cbar = fig.colorbar(c, ax=ax, orientation="horizontal", pad=0.14)
    cbar.set_ticks(np.arange(0, 2.2, 0.2))
    cbar.ax.get_yaxis().labelpad = 0
    cbar.set_label('Log$_{10}$ Precipitation Rate [mm day$^{-1}$]', rotation=0, fontdict={'size': 18, 'weight': 'bold'})
    for tick in cbar.ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(length=10, direction='in')

    # Save figure #
    if save_fig_boolean:
        plt.savefig(figure_path_and_name, dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.0, metadata=None)


def plot_undilute_B_L_VS_dilution_composites_no_vectors(binned_undilute_B_L_dilution_dataset, min_number_of_obs,
                                                        save_fig_boolean=False, figure_path_and_name='untitled.png'):
    """
    Composite plots of undilute B_L vs dilution with precipitation and no vectors.
    :param binned_undilute_B_L_dilution_dataset:
    :param min_number_of_obs:
    :param save_fig_boolean:
    :param figure_path_and_name:
    :return:
    """
    # Load data
    bin_number_of_samples = binned_undilute_B_L_dilution_dataset['bin_number_of_samples']
    bin_mean_precipitation_rate = binned_undilute_B_L_dilution_dataset['bin_mean_precipitation_rate']

    bin_mean_delta_dilution_leading = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_leading']
    bin_mean_delta_undilute_B_L_leading = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_leading']
    bin_number_pos_delta_dilution_leading = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_dilution_leading']
    bin_number_pos_delta_undilute_B_L_leading = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_undilute_B_L_leading']

    bin_mean_delta_dilution_lagging = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_lagging']
    bin_mean_delta_undilute_B_L_lagging = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_lagging']
    bin_number_pos_delta_dilution_lagging = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_dilution_lagging']
    bin_number_pos_delta_undilute_B_L_lagging = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_undilute_B_L_lagging']

    bin_mean_delta_dilution_centered = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_centered']
    bin_mean_delta_undilute_B_L_centered = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_centered']
    bin_number_pos_delta_dilution_centered = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_dilution_centered']
    bin_number_pos_delta_undilute_B_L_centered = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_undilute_B_L_centered']

    # Create mask for regions with insufficient obs #
    insufficient_obs_mask = bin_number_of_samples < min_number_of_obs

    n_bin = 200
    cmap_name = 'colors'
    colormap_colors = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

    # Create "centered" figure #
    fig = plt.figure(figsize=(10, 10))

    # Ask for, out of a 1x1 grid, the first axes #
    x_axis_limit = (-0.2, 0)
    x_tick_spacing = 0.04
    y_axis_limit = (-0.1, 0.1)
    y_tick_spacing = 0.04

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Dilution of B$_{L}$ [m s$^{-2}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax.set_ylabel('Undilute B$_{L}$  [m s$^{-2}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax.set(xlim=(x_axis_limit[0], x_axis_limit[1]), ylim=(y_axis_limit[0], y_axis_limit[1]))

    # Axis Ticks #
    plt.xticks(np.arange(x_axis_limit[0], x_axis_limit[1] + x_tick_spacing, x_tick_spacing))
    plt.yticks(np.arange(y_axis_limit[0], y_axis_limit[1] + y_tick_spacing, y_tick_spacing))

    ax.tick_params(axis="x", direction="in", length=12, width=3, color="black")
    ax.tick_params(axis="y", direction="in", length=12, width=3, color="black")

    ax.tick_params(axis="x", labelsize=18, labelrotation=0, labelcolor="black")
    ax.tick_params(axis="y", labelsize=18, labelrotation=0, labelcolor="black")

    for tick in ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    # Create "meshgrid" for contour plotting #
    BV1_bin_midpoint_meshgrid, BV2_bin_midpoint_meshgrid = np.meshgrid(bin_number_of_samples.BV1_bin_midpoint,
                                                                       bin_number_of_samples.BV2_bin_midpoint)

    BV1_bin_midpoint_meshgrid_DA = bin_number_of_samples.copy()
    BV1_bin_midpoint_meshgrid_DA.values = BV1_bin_midpoint_meshgrid

    BV2_bin_midpoint_meshgrid_DA = bin_number_of_samples.copy()
    BV2_bin_midpoint_meshgrid_DA.values = BV2_bin_midpoint_meshgrid

    # Contourf bin mean precipitation rate#

    c = ax.contourf(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    bin_mean_precipitation_rate.where(~insufficient_obs_mask), levels=np.arange(0, 21, 1),
                    cmap=colormap_colors, vmin=0.0, vmax=20.0, extend='max')

    # Contour bin mean B_L #
    c2 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=np.arange(-1, 0, 0.05), colors='k', linestyles='dashed', linewidths=3)

    c3 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=0, colors='k', linestyles='solid', linewidths=6)

    c4 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=np.arange(0.1, 2.1, 0.05), colors='k', linestyles='solid', linewidths=3)

    # Speckle regions with insufficient observations #
    ax.plot(BV1_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask),
            BV2_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask), 'ko', ms=1);

    # Mark the mode of observations #
    ax.plot(BV1_bin_midpoint_meshgrid_DA.where(bin_number_of_samples == bin_number_of_samples.max()),
            BV2_bin_midpoint_meshgrid_DA.where(bin_number_of_samples == bin_number_of_samples.max()), 'ks', ms=20,
            color=[0.7, 0.7, 0.7], fillstyle='full', markeredgecolor='k', markeredgewidth=3);

    # Colorbar #
    cbar = fig.colorbar(c, ax=ax, orientation="horizontal", pad=0.14)
    cbar.set_ticks(np.arange(0, 22, 2))
    cbar.ax.get_yaxis().labelpad = 0
    cbar.set_label('Precipitation Rate [mm day$^{-1}$]', rotation=0, fontdict={'size': 18, 'weight': 'bold'})
    for tick in cbar.ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(length=10, direction='in')

    # Save figure #
    if save_fig_boolean:
        plt.savefig(figure_path_and_name, dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.0, metadata=None)


def plot_undilute_B_L_VS_dilution_composites_log_precipitation_no_vectors(binned_undilute_B_L_dilution_dataset,
                                                                          min_number_of_obs, save_fig_boolean=False,
                                                                          figure_path_and_name='untitled.png'):
    """
    Composite plots of undilute B_L vs dilution with log precipitation and no vectors.
    :param binned_undilute_B_L_dilution_dataset:
    :param min_number_of_obs:
    :param save_fig_boolean:
    :param figure_path_and_name:
    :return:
    """
    # Load data
    bin_number_of_samples = binned_undilute_B_L_dilution_dataset['bin_number_of_samples']
    bin_mean_precipitation_rate = binned_undilute_B_L_dilution_dataset['bin_mean_precipitation_rate']

    bin_mean_delta_dilution_leading = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_leading']
    bin_mean_delta_undilute_B_L_leading = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_leading']
    bin_number_pos_delta_dilution_leading = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_dilution_leading']
    bin_number_pos_delta_undilute_B_L_leading = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_undilute_B_L_leading']

    bin_mean_delta_dilution_lagging = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_lagging']
    bin_mean_delta_undilute_B_L_lagging = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_lagging']
    bin_number_pos_delta_dilution_lagging = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_dilution_lagging']
    bin_number_pos_delta_undilute_B_L_lagging = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_undilute_B_L_lagging']

    bin_mean_delta_dilution_centered = binned_undilute_B_L_dilution_dataset['bin_mean_delta_dilution_centered']
    bin_mean_delta_undilute_B_L_centered = binned_undilute_B_L_dilution_dataset['bin_mean_delta_undilute_B_L_centered']
    bin_number_pos_delta_dilution_centered = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_dilution_centered']
    bin_number_pos_delta_undilute_B_L_centered = binned_undilute_B_L_dilution_dataset[
        'bin_number_pos_delta_undilute_B_L_centered']

    # Create mask for regions with insufficient obs #
    insufficient_obs_mask = bin_number_of_samples < min_number_of_obs

    n_bin = 200
    cmap_name = 'colors'
    colormap_colors = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

    # Create "centered" figure #
    fig = plt.figure(figsize=(10, 10))

    # Ask for, out of a 1x1 grid, the first axes #
    x_axis_limit = (-0.2, 0)
    x_tick_spacing = 0.04
    y_axis_limit = (-0.1, 0.1)
    y_tick_spacing = 0.04

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Dilution of B$_{L}$ [m s$^{-2}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax.set_ylabel('Undilute B$_{L}$  [m s$^{-2}$]', fontdict={'size': 24, 'weight': 'bold'})
    ax.set(xlim=(x_axis_limit[0], x_axis_limit[1]), ylim=(y_axis_limit[0], y_axis_limit[1]))

    # Axis Ticks #
    plt.xticks(np.arange(x_axis_limit[0], x_axis_limit[1] + x_tick_spacing, x_tick_spacing))
    plt.yticks(np.arange(y_axis_limit[0], y_axis_limit[1] + y_tick_spacing, y_tick_spacing))

    ax.tick_params(axis="x", direction="in", length=12, width=3, color="black")
    ax.tick_params(axis="y", direction="in", length=12, width=3, color="black")

    ax.tick_params(axis="x", labelsize=18, labelrotation=0, labelcolor="black")
    ax.tick_params(axis="y", labelsize=18, labelrotation=0, labelcolor="black")

    for tick in ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')

    # Create "meshgrid" for contour plotting #
    BV1_bin_midpoint_meshgrid, BV2_bin_midpoint_meshgrid = np.meshgrid(bin_number_of_samples.BV1_bin_midpoint,
                                                                       bin_number_of_samples.BV2_bin_midpoint)

    BV1_bin_midpoint_meshgrid_DA = bin_number_of_samples.copy()
    BV1_bin_midpoint_meshgrid_DA.values = BV1_bin_midpoint_meshgrid

    BV2_bin_midpoint_meshgrid_DA = bin_number_of_samples.copy()
    BV2_bin_midpoint_meshgrid_DA.values = BV2_bin_midpoint_meshgrid

    # Contourf bin mean precipitation rate#
    c = ax.contourf(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    np.log10(bin_mean_precipitation_rate.where(~insufficient_obs_mask)), levels=np.arange(0, 2.1, 0.1),
                    cmap=colormap_colors, vmin=0, vmax=2.0, extend='both')

    # Contour bin mean B_L #
    c2 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=np.arange(-1, 0, 0.05), colors='k', linestyles='dashed', linewidths=3)

    c3 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=0, colors='k', linestyles='solid', linewidths=6)

    c4 = ax.contour(BV1_bin_midpoint_meshgrid_DA, BV2_bin_midpoint_meshgrid_DA,
                    (BV1_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask) +
                     BV2_bin_midpoint_meshgrid_DA.where(~insufficient_obs_mask)),
                    levels=np.arange(0.1, 2.1, 0.05), colors='k', linestyles='solid', linewidths=3)

    # Speckle regions with insufficient observations #
    ax.plot(BV1_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask),
            BV2_bin_midpoint_meshgrid_DA.where(insufficient_obs_mask), 'ko', ms=1)

    # Mark the mode of observations #
    ax.plot(BV1_bin_midpoint_meshgrid_DA.where(bin_number_of_samples == bin_number_of_samples.max()),
            BV2_bin_midpoint_meshgrid_DA.where(bin_number_of_samples == bin_number_of_samples.max()), 'ks', ms=20,
            color=[0.7, 0.7, 0.7], fillstyle='full', markeredgecolor='k', markeredgewidth=3)

    # Colorbar #
    cbar = fig.colorbar(c, ax=ax, orientation="horizontal", pad=0.14)
    cbar.set_ticks(np.arange(0, 2.2, 0.2))
    cbar.ax.get_yaxis().labelpad = 0
    cbar.set_label('Log$_{10}$ Precipitation Rate [mm day$^{-1}$]', rotation=0, fontdict={'size': 18, 'weight': 'bold'})
    for tick in cbar.ax.xaxis.get_majorticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight('bold')
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(length=10, direction='in')

    # Save figure #
    if save_fig_boolean:
        plt.savefig(figure_path_and_name, dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.0, metadata=None)