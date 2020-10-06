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
# Output file name for datasets string list
ofile_datasets_string_list = ['ERAI_B_L_1p0_1x'] # TRMM and ERAi
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

    Q = Q.rename({'lev': 'level'})

    # Actually load data #
    land_sea_mask.load()
    PS.load()
    Q.load()
    T.load()

    # Clean up environment #
    gc.collect()

    # rename land_sea_mask
    landfrac = land_sea_mask
    landfrac = landfrac.rename({'land_sea_mask', 'landfrac'})

    print("Calculating true model pressure")
    true_pressure_midpoint, true_pressure_interface = mcc.calculate_true_pressure_model_pressure_midpoints_interfaces_pl(
        Q['level'] * 100., Q['time'], Q['level'], Q['lat'], Q['lon'], PS)

    print("Calculating saturation specific humidity")
    saturation_specific_humidity = xr.apply_ufunc(mcc.calculate_saturation_specific_humidity, true_pressure_midpoint, T,
                                                  output_dtypes=[Q.dtype])
    # Clean up environment #
    gc.collect();

    saturation_deficit = saturation_specific_humidity - Q

    ME = xr.apply_ufunc(mcc.calculate_moist_enthalpy, T, Q, output_dtypes=[T.dtype])
    ME_saturation = xr.apply_ufunc(mcc.calculate_saturation_moist_enthalpy, T, saturation_specific_humidity,
                                   output_dtypes=[T.dtype])
    # Clean up environment #
    del T
    gc.collect()

    print('Mass Weighted Averaging')
    _, _, mwa_ME_surface_to_850 = mcc.mass_weighted_vertical_integral_w_nan(ME, true_pressure_midpoint,
                                                                        true_pressure_interface, PS, 85000)

    _, _, mwa_ME_saturation_850_to_500 = mcc.mass_weighted_vertical_integral_w_nan(ME_saturation,
                                                                               true_pressure_midpoint,
                                                                               true_pressure_interface, 85000,
                                                                               50000)

    _, _, mwa_saturation_deficit_850_to_500 = mcc.mass_weighted_vertical_integral_w_nan(saturation_deficit,
                                                                                    true_pressure_midpoint,
                                                                                    true_pressure_interface, 85000,
                                                                                    50000)
    # Clean up environment #
    del Q, saturation_specific_humidity, saturation_deficit, ME, ME_saturation
    gc.collect()

    #################################
    ####  Output Data as NetCDF  ####
    #################################
    mcc.output_B_L(fname_datasets[0] + '_' + current_year_string + '.nc', landfrac, mwa_ME_surface_to_850,
                   mwa_ME_saturation_850_to_500, mwa_saturation_deficit_850_to_500)
