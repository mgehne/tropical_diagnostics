import numpy as np
import xarray as xr
from diagnostics import moisture_convection_coupling as mcc
from diagnostics import moisture_convection_coupling_plot as mccp
import gc

# Years to analyze
start_year = 2015
end_year = 2015

#########################
###  file names and paths  ###
#########################

# Precipitation
input_file_string_list_precipitation_rate = \
    ['/data/mgehne/Precip/precip.trmm.1x.1p0.nlat180.v7a.fillmiss.comp.1998-201806.nc'] # TRMM

# Atmosphere
input_file_string_list_specific_humidity = \
    ['/data/mgehne/ERAI/MetricsObs/CSF/shum.erai.an.pl.1p0.daily.201512-201603.nc']  # Specific humidity
input_file_string_list_temperature = \
    ['/data/mgehne/ERAI/MetricsObs/CSF/temp.erai.an.pl.1p0.daily.201512-201603.nc']  # Temperature
input_file_string_list_surface_pressure = \
    ['/data/mgehne/ERAI/MetricsObs/CSF/surface_pres.erai.an.sfc.1p0.daily.201512-201603.nc']  # Surface Pressure
# Land
input_file_string_list_land_frac = ['/data/mgehne/ERAI/MetricsObs/CSF/land_sea_mask.erai.1p0.nc']  # Land Fraction

# Output directory
odir_datasets_string_list = ['/data/mgehne/CSF_precipitation_analysis/']  # TRMM and ERAi
# Output file name for data sets string list
ofile_datasets_string_list = ['ERAI_B_L_1p0_1x'] # TRMM and ERAi
fname_datasets = [odir_datasets_string_list[i] +
                  ofile_datasets_string_list[i] for i in range(len(odir_datasets_string_list))]
# outut file names for binned data sets
bin_ofile_datasets_string_list = ['daily_TRMM_ERAI']
bin_fname_datasets = [odir_datasets_string_list[i] +
                  bin_ofile_datasets_string_list[i] for i in range(len(odir_datasets_string_list))]


print('input file = ' + input_file_string_list_specific_humidity[0])
print('input file = ' + input_file_string_list_temperature[0])
print('input file = ' + input_file_string_list_surface_pressure[0])
print('output dataset file = ' + fname_datasets[0])

for year in range(start_year, end_year + 1):
    print(year)
    # Define year strings #
    previous_year_string = str(year - 1)
    current_year_string = str(year)
    next_year_string = str(year + 1)
    while len(previous_year_string) < 4:
        previous_year_string = '0' + previous_year_string
    while len(current_year_string) < 4:
        current_year_string = '0' + current_year_string
    while len(next_year_string) < 4:
        next_year_string = '0' + next_year_string
#
    # Data is "lazy loaded", nothing is actually loaded until we "look" at data in some way #
    dataset_precipitation_rate = xr.open_dataset(input_file_string_list_precipitation_rate[0])
    dataset_specific_humidity = xr.open_dataset(input_file_string_list_specific_humidity[0])
    dataset_temperature = xr.open_dataset(input_file_string_list_temperature[0])
    dataset_surface_pressure = xr.open_dataset(input_file_string_list_surface_pressure[0])
    dataset_land = xr.open_dataset(input_file_string_list_land_frac[0])

    # Make data arrays, loading only the year of interest #
    full_lat = dataset_surface_pressure['lat']
    full_lon = dataset_surface_pressure['lon']
    land_sea_mask = dataset_land['land_sea_mask']

    PS = dataset_surface_pressure['surface_pres'].sel(
        time=slice(current_year_string + '-12-01', next_year_string + '-03-31'), lat=slice(-15, 15))  # [Pa]
    Q = dataset_specific_humidity['shum'].sel(
        time=slice(current_year_string + '-12-01', next_year_string + '-03-31'), lat=slice(-15, 15),
        level=slice(70, 1000))  # [Kg/Kg]
    T = dataset_temperature['temp'].sel(time=slice(current_year_string + '-12-01', next_year_string + '-03-31'),
                                        lat=slice(-15, 15), level=slice(70, 1000))  # [K]
    precipitation_rate = dataset_precipitation_rate['precip'].sel(
        time=slice(current_year_string + '-12-01', next_year_string + '-03-31'), lat=slice(-15, 15)) * 24
    # Currently [mm/hr]. Convert to [mm/day]

    # Actually load data #
    land_sea_mask.load()
    PS.load()
    Q.load()
    T.load()
    precipitation_rate.load()

    # Clean up environment #
    gc.collect()

    # rename land_sea_mask
    landfrac = land_sea_mask
    landfrac = landfrac.rename({'land_sea_mask', 'landfrac'})

    # compute vertical averages need for B_L computation
    mwa_ME_surface_to_850, mwa_ME_saturation_850_to_500, mwa_saturation_deficit_850_to_500 = \
        mcc.compute_mwa_ME_components(Q, T, PS)

    # write vertical averages to netcdf
    mcc.output_mwa(fname_datasets[0] + '_' + current_year_string + '.nc', landfrac, mwa_ME_surface_to_850,
                   mwa_ME_saturation_850_to_500, mwa_saturation_deficit_850_to_500)

    print('mask out land points for binning')
    landfrac = landfrac.sel(lat=slice(-15, 15))
    is_valid_ocean_mask = (landfrac < 0.1)
    precipitation_rate = precipitation_rate.where(is_valid_ocean_mask, other=np.nan)
    mwa_ME_surface_to_850 = mwa_ME_surface_to_850.where(is_valid_ocean_mask, other=np.nan)
    mwa_ME_saturation_850_to_500 = mwa_ME_saturation_850_to_500.where(is_valid_ocean_mask, other=np.nan)
    mwa_saturation_deficit_850_to_500 = mwa_saturation_deficit_850_to_500.where(is_valid_ocean_mask, other=np.nan)

    precipitation_rate = precipitation_rate.resample(time='1D').mean('time')
    mwa_ME_surface_to_850 = mwa_ME_surface_to_850.resample(time='1D').mean('time')
    mwa_ME_saturation_850_to_500 = mwa_ME_saturation_850_to_500.resample(time='1D').mean('time')
    mwa_saturation_deficit_850_to_500 = mwa_saturation_deficit_850_to_500.resample(time='1D').mean('time')

    print(precipitation_rate.shape)

    print('Calculating B_L and Components')
    B_L, undilute_B_L, dilution_of_B_L = mcc.compute_B_L(mwa_ME_surface_to_850, mwa_ME_saturation_850_to_500,
                                                         mwa_saturation_deficit_850_to_500)
    print(B_L.shape)

    mcc.calculate_undilute_B_L_dilution_binned_composites(precipitation_rate, B_L, undilute_B_L, dilution_of_B_L,
                                                          year, bin_fname_datasets[0])


########## plot results
start_year = (2015, 2015)
end_year = (2015, 2015)

input_file_string_list = ['/data/mgehne/CSF_precipitation_analysis/daily_TRMM_ERAI']
# Output directory for figures string list
odir_figures_string_list = ['/data/mgehne/CSF_precipitation_analysis/Plots/B_L_analysis/']
# output file name
ofile_figures_string_list = ['daily_' + str(start_year[0]) + '_' + str(end_year[0]) + '_TRMM_ERAI.png']
fname_figures = [odir_figures_string_list[i] + ofile_figures_string_list[i] for i in range(len(odir_figures_string_list))]

min_number_of_obs = 200

print('input file = ' + input_file_string_list[0])
print('output figure directory = ' + odir_figures_string_list[0])

# Define paths of files we wish to load #
paths_all_years_B_L_binned = input_file_string_list[0] + '_B_L_binned_precipitation_rate_2015.nc'

paths_all_years_undilute_B_L_dilution_binned = input_file_string_list[0] + '_undilute_B_L_dilution_binned_data_2015.nc'

#### Limit files to years of interest   ###
year_limited_paths_B_L_binned = paths_all_years_B_L_binned
year_limited_paths_undilute_B_L_dilution_binned = paths_all_years_undilute_B_L_dilution_binned

for year in range(start_year[0], end_year[0] + 1):

    # Define year strings #
    current_year_string = str(year)
    while len(current_year_string) < 4:
        current_year_string = '0' + current_year_string

    # load datasets of binned composites
    B_L_binned_precipitation_rate_dataset = mcc.process_binned_B_L_dataset(year_limited_paths_B_L_binned)

    binned_undilute_B_L_dilution_dataset = mcc.process_binned_undilute_B_L_dilution_dataset(
        year_limited_paths_undilute_B_L_dilution_binned)

    # Plot B_L binned precipitation rate
    figure_path_and_name = odir_figures_string_list[0] + 'B_L_binned_precipitation_rate_composite_' + \
                           ofile_figures_string_list[0]
    print(figure_path_and_name)
    mccp.plot_B_L_binned_precipitation_rate(B_L_binned_precipitation_rate_dataset, min_number_of_obs, True,
                                            figure_path_and_name)

    # Plot undilute B_L vs dilution composites
    figure_path_and_name = odir_figures_string_list[0] + 'undilute_B_L_vs_dilution_composite_' + \
                           ofile_figures_string_list[0]
    print(figure_path_and_name)
    mccp.plot_undilute_B_L_VS_dilution_composites_V1(binned_undilute_B_L_dilution_dataset, min_number_of_obs, True,
                                                figure_path_and_name)

    # Plot undilute B_L vs dilution composites with log precipitation scale
    figure_path_and_name = odir_figures_string_list[0] + 'undilute_B_L_vs_dilution_composite_log_precipitation_' + \
                           ofile_figures_string_list[0]
    print(figure_path_and_name)
    mccp.plot_undilute_B_L_VS_dilution_composites_log_precipitation(binned_undilute_B_L_dilution_dataset,
                                                                    min_number_of_obs, True, figure_path_and_name)

    # Plot undilute B_L vs dilution composites zoom out
    figure_path_and_name = odir_figures_string_list[0] + 'undilute_B_L_vs_dilution_composite_ZO_' + \
                           ofile_figures_string_list[0]
    print(figure_path_and_name)
    mccp.plot_undilute_B_L_VS_dilution_composites_V1_zoom_out(binned_undilute_B_L_dilution_dataset,
                                                              min_number_of_obs, True, figure_path_and_name)

    # Plot undilute B_L vs dilution composites with log precipitation scale zoom out
    figure_path_and_name = odir_figures_string_list[0] + 'undilute_B_L_vs_dilution_composite_ZO_log_precipitation_' + \
                           ofile_figures_string_list[0]
    print(figure_path_and_name)
    mccp.plot_undilute_B_L_VS_dilution_composites_log_precipitation_zoom_out(binned_undilute_B_L_dilution_dataset,
                                                                             min_number_of_obs, True, figure_path_and_name)

    # Plot undilute B_L vs dilution composites no vectors
    figure_path_and_name = odir_figures_string_list[0] + 'undilute_B_L_vs_dilution_composite_no_vectors_' + \
                           ofile_figures_string_list[0]
    print(figure_path_and_name)
    mccp.plot_undilute_B_L_VS_dilution_composites_no_vectors(binned_undilute_B_L_dilution_dataset,
                                                             min_number_of_obs, True, figure_path_and_name)

    # Plot undilute B_L vs dilution composites with log precipitation scale no vectors
    figure_path_and_name = odir_figures_string_list[0] + \
                           'undilute_B_L_vs_dilution_composite_log_precipitation_no_vectors_' + \
                           ofile_figures_string_list[0]
    print(figure_path_and_name)
    mccp.plot_undilute_B_L_VS_dilution_composites_log_precipitation_no_vectors(binned_undilute_B_L_dilution_dataset,
                                                                               min_number_of_obs, True,
                                                                               figure_path_and_name)
