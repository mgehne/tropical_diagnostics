import numpy as np
import xarray as xr
from tropical_diagnostics import moisture_convection_coupling as mcc
from tropical_diagnostics import moisture_convection_coupling_plot as mccp

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
# Output file name for datasets string list
ofile_datasets_string_list = ['TRMM_ERAi_1p0_1x'] # TRMM and ERAi
# Define output string for datasets and figures
fname_datasets = [odir_datasets_string_list[i] +
                  ofile_datasets_string_list[i] for i in range(len(odir_datasets_string_list))]

# output directory for figures
odir_figures_string_list = ['/data/mgehne/CSF_precipitation_analysis/Plots/']  # TRMM and ERAi
# Output file name for figures
ofile_figures_string_list = ofile_datasets_string_list
# Define output string for figures
fname_figures = [odir_figures_string_list[i] + ofile_figures_string_list[i] for i in
                 range(len(odir_figures_string_list))]


print('input file = ' + input_file_string_list_precipitation_rate[0])
print('output dataset file = ' + fname_datasets[0])

#############
# Limit files to the time interval of interest.
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

    #  Load ERAi Data
    # Data is "lazy loaded", nothing is actually loaded until we "look" at data in some way #
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
    Q = dataset_specific_humidity['shum'].sel(time=slice(current_year_string + '-12-01', next_year_string + '-03-31'),
                                              lat=slice(-15, 15), level=slice(70, 1000))  # [Kg/Kg]
    T = dataset_temperature['temp'].sel(time=slice(current_year_string + '-12-01', next_year_string + '-03-31'),
                                        lat=slice(-15, 15), level=slice(70, 1000))  # [K]

    # Actually load data #
    land_sea_mask.load()
    PS.load()
    Q.load()
    T.load()

    print("Modifying landfrac as needed")
    landfrac = land_sea_mask
    landfrac = landfrac.rename({'land_sea_mask', 'landfrac'})
    landfrac = landfrac.sortby('lat', ascending=True)
    landfrac = landfrac.sel(lat=slice(-15, 15))

    #  Load Precipitation Data
    dataset_precipitation_rate = xr.open_dataset(input_file_string_list_precipitation_rate[0])
    # Currently [mm/hr] Convert to [mm/day]
    precipitation_rate = dataset_precipitation_rate['precip'].sel(
        time=slice(current_year_string + '-12-01', next_year_string + '-03-31'), lat=slice(-15, 15)) * 24
    precipitation_rate.load()

    #  Resample data to daily to match 'time' coordinates
    PS = PS.resample(time='1D').mean('time')
    Q = Q.resample(time='1D').mean('time')
    T = T.resample(time='1D').mean('time')
    precipitation_rate = precipitation_rate.resample(time='1D').mean('time')

    #  Limit to Oceanic (<10% Land) Points
    print('Applying Land/Ocean Mask')
    is_valid_ocean_mask = (landfrac < 0.1)

    # Apply ocean mask to appropriate variables, setting invalid locations to nan #
    PS = PS.where(is_valid_ocean_mask, other=np.nan)
    Q = Q.where(is_valid_ocean_mask, other=np.nan)
    T = T.where(is_valid_ocean_mask, other=np.nan)
    precipitation_rate = precipitation_rate.where(is_valid_ocean_mask, other=np.nan)

    #####################################################
    ####  USE THIS METHOD FOR DATA ON MODEL LEVELS.  ####
    ####  Calculate True Model Pressure              ####
    #####################################################

    # print("Calculating true model pressure")
    # true_pressure_midpoint,true_pressure_interface =
    # mcc.calculate_true_pressure_model_pressure_midpoints_interfaces_ml(hyam, hybm, hyai, hybi, P0, PS)

    ################################################################
    ####  USE THIS METHOD FOR DATA ALREADY ON PRESSURE LEVELS.  ####
    ####. Calculate True Model Pressure                         ####
    ################################################################

    print("Calculating true model pressure")
    true_pressure_midpoint = Q['level'] * 100.  # To convert to Pa
    true_pressure_midpoint, true_pressure_interface = \
        mcc.calculate_true_pressure_model_pressure_midpoints_interfaces_pl(true_pressure_midpoint,
                                                                           Q['time'], Q['level'], Q['lat'], Q['lon'],
                                                                           PS)

    #  Calculate Saturation Specific Humidity
    print("Calculating saturation specific humidity")
    saturation_specific_humidity = xr.apply_ufunc(mcc.calculate_saturation_specific_humidity, true_pressure_midpoint, T,
                                                  output_dtypes=[Q.dtype])

    #  Column Integrate Variables
    upper_level_integration_limit_Pa = 10000  # [Pa]
    lower_level_integration_limit_Pa = 100000  # [Pa]

    print('Column Integrating')
    ci_q, _, _ = mcc.mass_weighted_vertical_integral_w_nan(Q, true_pressure_midpoint, true_pressure_interface,
                                                           lower_level_integration_limit_Pa,
                                                           upper_level_integration_limit_Pa)

    ci_q_sat, _, _ = mcc.mass_weighted_vertical_integral_w_nan(saturation_specific_humidity, true_pressure_midpoint,
                                                               true_pressure_interface,
                                                               lower_level_integration_limit_Pa,
                                                               upper_level_integration_limit_Pa)
    #  column  saturation fraction
    csf = ci_q / ci_q_sat

    # Calculate CSF Precipitation Rate Composites and save to file
    mcc.calculate_csf_precipitation_binned_composites(csf, precipitation_rate, year, fname_datasets[0], 'time')

#######################################################
####  Plot composites ####
#######################################################
# which experiment or model run are we using as verification
verification_simulation_number = 0
# minimum number of obs per bin to include bin in plotting
min_number_of_obs = 200

for simulation_number in [0]:  # range(len(fname_datasets)):

    # Load saved variables
    # using lists of files to load requires dask
    # list_of_files = glob(fname_datasets[simulation_number] + '*' + 'CSF_binned_precipitation_rate' + '*')
    # list_of_files = glob(fname_datasets[simulation_number] + '*' + 'CSF_precipitation_binned_data' + '*')

    list_of_files = fname_datasets[simulation_number] + '_CSF_binned_precipitation_rate' + '_2015.nc'
    csf_binned_precipitation_rate_dataset = mcc.process_binned_single_variable_dataset(list_of_files)

    list_of_files = fname_datasets[simulation_number] + '_CSF_precipitation_binned_data' + '_2015.nc'
    binned_CSF_precipitation_dataset = mcc.process_binned_csf_precipitation_rate_dataset(list_of_files)

    #  Plotting CSF Binned Precipitation Rate Figures
    save_fig_boolean = True
    figure_path_and_name = fname_figures[simulation_number] + '_CSF_binned_precipitation_rate.png'
    print(figure_path_and_name)
    mccp.plot_csf_binned_precipitation_rate(csf_binned_precipitation_rate_dataset, min_number_of_obs, save_fig_boolean,
                                            figure_path_and_name)

    #  Plotting Coevolution Figures
    save_fig_boolean = True
    figure_path_and_name = fname_figures[simulation_number] + '_center_diff_CSF_precipitation_coevolution.png'
    print(figure_path_and_name)
    mccp.plot_CSF_precipitation_rate_composites(binned_CSF_precipitation_dataset, min_number_of_obs, save_fig_boolean,
                                                figure_path_and_name)

    if simulation_number != verification_simulation_number:
        save_fig_boolean = True
        # Load verification dataset
        difference_figure_path_and_name = odir_figures_string_list[simulation_number] + \
                                          ofile_figures_string_list[simulation_number] + '_minus_' + \
                                          ofile_figures_string_list[verification_simulation_number] + \
                                          '_center_diff_CSF_precipitation_coevolution.png'
        print(difference_figure_path_and_name)

        # loading a list of files requires dask
        # list_of_files = glob(fname_datasets[verification_simulation_number] + '*' +
        #                      'CSF_precipitation_binned_data' + '*')
        list_of_files = fname_datasets[verification_simulation_number] + '_CSF_precipitation_binned_data' + '_2015.nc'
        binned_CSF_precipitation_dataset_verificaton = \
            mcc.process_binned_csf_precipitation_rate_dataset(list_of_files)

        mccp.plot_CSF_precipitation_rate_difference_composites(binned_CSF_precipitation_dataset_verificaton,
                                                               binned_CSF_precipitation_dataset, min_number_of_obs,
                                                               save_fig_boolean, difference_figure_path_and_name)
