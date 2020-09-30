import numpy as np
import xarray as xr
from diagnostics import moisture_convection_coupling as mcc
import gc

# Years to analyze
start_year = 2015
end_year = 2016

#########################
###.  ERAi and TRMM.  ###
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

#############

print('input file = ' + input_file_string_list_precipitation_rate[0])
print('output dataset file = ' + fname_datasets[0])

#########################################
# Define paths of files we wish to load #
#########################################

# Limit files to the year of interest, and the year on either side. Given the file naming convention, the
# year prior/after may have dates from the year of interest in the file

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

    ##########################
    ####  Load ERAi Data  ####
    ##########################

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

    # Clean up environment #
    gc.collect()

    ###############################################
    ####  Modify "landfrac" Variable as Needed ####
    ###############################################

    print("Modifying landfrac as needed")

    # landfrac = land_sea_mask.rename({'Latitude':'lat','Longitude':'lon'})
    landfrac = land_sea_mask
    landfrac = landfrac.rename({'land_sea_mask', 'landfrac'})
    print(landfrac)

    # The landfrac variable does not have lat/lon coordinates. Assign those of variables and check to make sure they make sense #

    # print(landfrac.coords['lat'])
    #landfrac.coords['lat'] = full_lat.coords['lat'] * -1
    #landfrac.coords['lon'] = full_lon.coords['lon']

    #landfrac = landfrac.transpose()
    landfrac = landfrac.sel(lat=slice(15, -15))
    landfrac = landfrac.sortby('lat', ascending=True)

    # Clean up environment #
    gc.collect()

    ##########################
    ####  Load Precipitation Data  ####
    ##########################

    dataset_precipitation_rate = xr.open_dataset(input_file_string_list_precipitation_rate[0])

    precipitation_rate = dataset_precipitation_rate['precip'].sel(
        time=slice(current_year_string + '-12-01', next_year_string + '-03-31'), lat=slice(-15, 15)) * (
                             24)  # Currently [mm/hr]. Convert to [mm/day]

    precipitation_rate.load()

    ##############################################################
    ####  Resample data to daily to match 'time' coordinates  ####
    ##############################################################

    PS = PS.resample(time='1D').mean('time')
    Q = Q.resample(time='1D').mean('time')
    T = T.resample(time='1D').mean('time')
    precipitation_rate = precipitation_rate.resample(time='1D').mean('time')

    ###############################################
    ####  Limit to Oceanic (<10% Land) Points  ####
    ###############################################

    print('Applying Land/Ocean Mask')

    # Create ocean mask #
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
    # Clean up environment #
    # gc.collect();

    ################################################################
    ####  USE THIS METHOD FOR DATA ALREADY ON PRESSURE LEVELS.  ####
    ####. Calculate True Model Pressure                         ####
    ################################################################

    print("Calculating true model pressure")

    # Set upper most interface equal to uppermost level midpoint, and lowest interface equal to surface pressure.
    # This will still permit the desired vertical integral, just choose appropriate upper and lower integration limits

    # Model level midpoint
    true_pressure_midpoint = Q['level'] * 100.  # To convert to Pa
    true_pressure_midpoint, true_pressure_interface = \
        mcc.calculate_true_pressure_model_pressure_midpoints_interfaces_pl(true_pressure_midpoint,
                                                                           Q['time'], Q['level'], Q['lat'], Q['lon'],
                                                                           PS)

    ##################################################
    ####  Calculate Saturation Specific Humidity  ####
    ##################################################

    print("Calculating saturation specific humidity")
    saturation_specific_humidity = xr.apply_ufunc(mcc.calculate_saturation_specific_humidity, true_pressure_midpoint, T,
                                                  output_dtypes=[Q.dtype])

    # Clean up environment #
    gc.collect();\

    ######################################
    ####  Column Integrate Variables  ####
    ######################################

    upper_level_integration_limit_Pa = 10000  # [Pa]
    lower_level_integration_limit_Pa = 100000  # [Pa]

    print('Column Integrating')

    ci_q, _, _ = mcc.mass_weighted_vertical_integral_w_nan(Q, true_pressure_midpoint, true_pressure_interface,
                                                           lower_level_integration_limit_Pa,
                                                           upper_level_integration_limit_Pa)
    # print(ci_q)
    # print(ci_q.min())
    # print(ci_q.max())
    # print(ci_q.mean())
    # plt.figure()
    # ci_q.isel(time = 0).plot()

    ci_q_sat, _, _ = mcc.mass_weighted_vertical_integral_w_nan(saturation_specific_humidity, true_pressure_midpoint,
                                                               true_pressure_interface,
                                                               lower_level_integration_limit_Pa,
                                                               upper_level_integration_limit_Pa)
    # print(ci_q_sat)
    # print(ci_q_sat.min())
    # print(ci_q_sat.max())
    # print(ci_q_sat.mean())
    # plt.figure()
    # ci_q_sat.isel(time = 0).plot()

    csf = ci_q / ci_q_sat
    # print(csf)
    # print(csf.min())
    # print(csf.max())
    # plt.figure()

    # Clean up environment #
    gc.collect()

    #######################################################
    ####  Calculate CSF Precipitation Rate Composites  ####
    #######################################################

    mcc.calculate_csf_precipitation_binned_composites(csf, precipitation_rate, year, fname_datasets[0])


