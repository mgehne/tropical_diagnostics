"""
Routines to compute column saturation fraction (CSF) and the relationship between precipitation and CSF.
Contributed by Brandon Wolding.

"""
import numpy as np
import xarray as xr


def calculate_true_pressure_model_pressure_midpoints_interfaces_ml(hyam, hybm, hyai, hybi, P0, PS):
    """
    Notes from Jim Benedict:
    To compute a pressure array, use the formula:

    p(k) = hyam(k)*P0 + hybm(k)*PS
    where p is the computed pressure (Pa), k is the level index, hyam and hybm are coefficients contained in the "h3" files,
    P0 is a single reference surface pressure (units Pa, also in h3 files), and PS (Pa) is the actual surface pressure
    (in h2 files). Note that the 1D "lev" arrays are only reference or nominal pressure levels they are similar to the
    actual pressures over the tropical oceans but differ greatly over mountains.

    In practice, you compute p(time, k,lat,lon) = hyam(k)*P0 + hybm(k)*PS(time,lat,lon) to obtain the pressure array,
    which you can then feed into your relative humidity (or vapor pressure) equation to get column saturation.

    :param hyam: model level a parameter for level midpoint
    :param hybm: model level b parameter for level midpoint
    :param hyai: model level a parameter for level interface
    :param hybi: model level b parameter for level interface
    :param P0: reference surface pressure
    :param PS: actual surface pressure
    :return: true_pressure_midpoint, true_pressure_interface
    """
    true_pressure_midpoint = (hyam * P0) + (hybm * PS)  # [Pa]
    true_pressure_interface = (hyai * P0) + (hybi * PS)  # [Pa]

    true_pressure_midpoint.attrs['units'] = 'Pa'
    true_pressure_interface.attrs['units'] = 'Pa'

    return true_pressure_midpoint, true_pressure_interface

def calculate_true_pressure_model_pressure_midpoints_interfaces_pl(true_pressure_midpoint, time, level, lat, lon, PS):
    """
    Compute true model pressure interfaces and midpoints for each level. Use this for data on pressure levels.
    Set upper most interface equal to uppermost level midpoint, and lowest interface equal to surface pressure.
    This will still permit the desired vertical integral, just choose appropriate upper and lower integration limits
    :param true_pressure_midpoint:
    :param time: time coordinate
    :param level: level coordinate
    :param lat: latitude coordinate
    :param lon: longitiude coordinate
    :param PS: actual surface pressure
    :return true_pressure_midpoint, true_pressure_interface:
    """
# Set upper most interface equal to uppermost level midpoint, and lowest interface equal to surface pressure.
    # This will still permit the desired vertical integral, just choose appropriate upper and lower integration limits

    # Model level midpoint
    #true_pressure_midpoint = Q['level'] * 100.  # To convert to Pa
    true_pressure_midpoint = true_pressure_midpoint.rename('true_pressure_midpoint_Pa')
    true_pressure_midpoint = true_pressure_midpoint.expand_dims({'lat': lat, 'lon': lon, 'time': time})
    true_pressure_midpoint = true_pressure_midpoint.transpose('time', 'level', 'lat', 'lon')

    # Model level interfaces
    true_pressure_interface = np.empty((len(time), len(lat), len(lon), len(level) + 1))

    for interface_level_counter in range(len(level) + 1):
        if interface_level_counter == 0:
            # Set upper most interface equal to uppermost level midpoint
            true_pressure_interface[:, :, :, interface_level_counter] = level.isel(level=0).values
        elif interface_level_counter == (len(level)):
            # Set lowest interface equal to surface pressure
            true_pressure_interface[:, :, :, interface_level_counter] = PS
        else:
            # Set middle interfaces equal to half way points between level midpoints
            true_pressure_interface[:, :, :, interface_level_counter] = \
                (level.isel(level=interface_level_counter - 1).values +
                 level.isel(level=interface_level_counter).values) / 2.

    coords = {'time': time, 'lat': lat, 'lon': lon, 'ilev': np.arange(1, len(level) + 2)}
    dims = ['time', 'lat', 'lon', 'ilev']
    true_pressure_interface = xr.DataArray(true_pressure_interface, dims=dims, coords=coords) * 100.  # To convert to Pa
    true_pressure_interface.attrs['units'] = 'Pa'

    true_pressure_interface = true_pressure_interface.transpose('time', 'ilev', 'lat', 'lon')

    return true_pressure_midpoint, true_pressure_interface


def calculate_saturation_specific_humidity(pressure, temperature):
    """
    Calculate Relative Humidity Based on Outdated WMO 1987 adaptation of Goff and Gratch Saturation Vapor Pressure (SVP)
    Units of pressure are [Pa], temperature are [K]
    :param pressure:
    :param temperature:
    :return: saturation_specific_humidity
    """

    t_0 = 273.16
    log10svp = 10.79574 * (1 - t_0 / temperature) - 5.028 * xr.ufuncs.log10(temperature / t_0) + 1.50475 * (
                10 ** -4) * (1 - 10 ** (-8.2969 * (temperature / (t_0 - 1)))) + 0.42873 * (10 ** -3) * (
                           10 ** (4.76955 * ((1 - t_0) / temperature))) + 0.78614 + 2.0

    svp = 10 ** log10svp
    eta = 0.62198  # Ratios of molecular weights of water and dry air
    saturation_specific_humidity = eta * svp / pressure

    return saturation_specific_humidity


def calculate_moist_enthalpy(temperature, specific_humidity):
    """
    Calculate the moist enthalpy from temperature and specific humidity.
    :param temperature: temperature array in units of K
    :param specific_humidity: specific humidity array in units of Kg Kg^-1
    :return: moist enthalpy array
    """
    # Define constants
    Cp = 1004.  # [J kg^-1 K^-1]
    Lv = 2.5 * 10. ** 6.  # [J kg^-1]
    # Calculate MSE
    ME = (Cp * temperature) + (Lv * specific_humidity)  # [J kg^-1]

    return ME


def calculate_saturation_moist_enthalpy(temperature, saturation_specific_humidity):
    """
    Calculate the saturation moist enthalpy from temperature and saturation specific humidity.
    :param temperature: temperature array in units of K
    :param saturation_specific_humidity: saturation specific humidity array in units of Kg Kg^-1
    :return: saturation moist enthalpy array
    """
    # Define constants
    Cp = 1004.  # [J kg^-1 K^-1]
    Lv = 2.5 * 10. ** 6.  # [J kg^-1]
    # Calculate MSE_sat
    ME_saturation = (Cp * temperature) + (Lv * saturation_specific_humidity)  # [J kg^-1]

    return ME_saturation


def mass_weighted_vertical_integral_w_nan(variable_to_integrate, pressure_model_level_midpoint_Pa,
                                          pressure_model_level_interface_Pa, max_pressure_integral_array_Pa,
                                          min_pressure_integral_array_Pa):
    """
    Column integrate a variable that has nan values. Accepts both integers and arrays as min and max pressure limits.
    Pressure arrays should be in units of pascals.
    :param variable_to_integrate: datra array of variable we want to column integrate
    :param pressure_model_level_midpoint_Pa: level midpoints
    :param pressure_model_level_interface_Pa: level interfaces
    :param max_pressure_integral_array_Pa: maximum pressure to integrate to
    :param min_pressure_integral_array_Pa: minimum pressure to integrate to
    :return ci_variable, dp_total, mwa_variable: column integrated variable, total pressure delta, mass weighted
    vertical integral of variable
    """
    # Define constants
    g = 9.8  # [m s^-2]

    # Set all model interfaces less than minimum pressure equal to minimum pressure, and more than
    # maximum pressure to maximum pressure. This way, when you calculate "dp", these layers will not have mass.
    pressure_model_level_interface_Pa = pressure_model_level_interface_Pa.where(
        pressure_model_level_interface_Pa < max_pressure_integral_array_Pa, other=max_pressure_integral_array_Pa)
    pressure_model_level_interface_Pa = pressure_model_level_interface_Pa.where(
        pressure_model_level_interface_Pa > min_pressure_integral_array_Pa, other=min_pressure_integral_array_Pa)

    # Calculate delta pressure for each model level
    dp = pressure_model_level_midpoint_Pa.copy()
    dp.values = xr.DataArray(pressure_model_level_interface_Pa.isel(
        ilev=slice(1, len(pressure_model_level_interface_Pa.ilev))).values - pressure_model_level_interface_Pa.isel(
        ilev=slice(0, -1)).values)  # Slice indexing is (inclusive start, exclusive stop)

    # Set dp = nan at levels missing data so mass of those levels not included in calculation of dp_total
    dp = dp.where(~xr.ufuncs.isnan(variable_to_integrate), drop=False, other=np.nan)

    # Mass weight each layer
    ci_variable = variable_to_integrate * dp / g

    # Integrate over levels
    ci_variable = ci_variable.sum('level', min_count=1)
    dp_total = dp.sum('level', min_count=1)

    # Set ci_variable to nan wherever dp_total is zero or nan
    ci_variable = ci_variable.where(~(dp_total == 0), drop=False, other=np.nan)
    ci_variable = ci_variable.where(~xr.ufuncs.isnan(dp_total), drop=False, other=np.nan)

    # Calculate mass weighted vertical average over layer integrated over
    mwa_variable = ci_variable * g / dp_total

    return ci_variable, dp_total, mwa_variable


def calculate_backward_forward_center_difference(variable_to_difference):
    """
    Calculate backwards, forwards and center differences of a variable. Input variable needs to have a time
    dimension to difference.
    :param variable_to_difference: input data array with time dimension
    :return backwards_differenced_variable, forwards_differenced_variable, center_differenced_variable:
    """

    first_time = variable_to_difference.isel(time=0)
    last_time = variable_to_difference.isel(time=-1)

    first_time.values = np.full(np.shape(first_time), np.nan)
    last_time.values = np.full(np.shape(last_time), np.nan)

    # Leading (backwards difference)
    backwards_differenced_variable = variable_to_difference.isel(time=slice(1, len(
        variable_to_difference.time) + 1)).copy()  # Careful to assign backwards differenced data to correct time step
    backwards_differenced_variable.values = variable_to_difference.isel(
        time=slice(1, len(variable_to_difference.time))).values - variable_to_difference.isel(
        time=slice(0, -1)).values  # Slice indexing is (inclusive start, exclusive stop)
    backwards_differenced_variable = xr.concat((first_time, backwards_differenced_variable), 'time')

    # Lagging (forwards difference)
    forwards_differenced_variable = variable_to_difference.isel(
        time=slice(0, -1)).copy()  # Careful to assign forwards differenced data to correct time step
    forwards_differenced_variable.values = variable_to_difference.isel(
        time=slice(1, len(variable_to_difference.time))).values - variable_to_difference.isel(time=slice(0, -1)).values
    forwards_differenced_variable = xr.concat((forwards_differenced_variable, last_time), 'time')

    # Centered (center difference)
    center_differenced_variable = variable_to_difference.isel(
        time=slice(1, -1)).copy()  # Careful to assign center differenced data to correct time step
    center_differenced_variable.values = variable_to_difference.isel(
        time=slice(2, len(variable_to_difference.time))).values - variable_to_difference.isel(time=slice(0, -2)).values
    center_differenced_variable = xr.concat((first_time, center_differenced_variable, last_time), 'time')

    return backwards_differenced_variable, forwards_differenced_variable, center_differenced_variable


def bin_by_one_variable(variable_to_be_binned, BV1, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector):
    """
    Bin one variable by another. Find the mean of the first input variable within bins of the second input variable.
    :param variable_to_be_binned: variable to find the mean in each bin of BV1 (e.g. precipitation)
    :param BV1: variable used to create the bins (e.g. column saturation fraction)
    :param lower_BV1_bin_limit_vector: lower bin limits of BV1
    :param upper_BV1_bin_limit_vector: upper bin limits of BV1
    :return bin_mean_variable, bin_number_of_samples: mean variable in each bin, number of samples in each bin
    """
    # Define bins
    BV1_bin_midpoint = (lower_BV1_bin_limit_vector + upper_BV1_bin_limit_vector) / 2

    lower_BV1_bin_limit_DA = xr.DataArray(lower_BV1_bin_limit_vector, coords=[BV1_bin_midpoint],
                                          dims=['BV1_bin_midpoint'])
    upper_BV1_bin_limit_DA = xr.DataArray(upper_BV1_bin_limit_vector, coords=[BV1_bin_midpoint],
                                          dims=['BV1_bin_midpoint'])

    number_of_BV1_bins = len(BV1_bin_midpoint)

    # Instantiate composite variable
    coords = {'BV1_bin_midpoint': BV1_bin_midpoint}
    dims = ['BV1_bin_midpoint']

    bin_number_of_samples = xr.DataArray(np.full(len(lower_BV1_bin_limit_DA), np.nan), dims=dims, coords=coords)

    # instantiate mean variable array
    bin_mean_variable = bin_number_of_samples.copy()

    # Calculate bin mean and number of positive values in each bin
    for BV1_bin in BV1_bin_midpoint:
        bin_index = (BV1 >= lower_BV1_bin_limit_DA.where(lower_BV1_bin_limit_DA.BV1_bin_midpoint == BV1_bin,
                                                         drop=True).values) & (
                                BV1 <= upper_BV1_bin_limit_DA.where(upper_BV1_bin_limit_DA.BV1_bin_midpoint == BV1_bin,
                                                                    drop=True).values)

        bin_number_of_samples.loc[dict(BV1_bin_midpoint=BV1_bin)] = bin_index.sum()

        if np.isfinite(variable_to_be_binned.where(bin_index)).sum() > 0:
            bin_mean_variable.loc[dict(BV1_bin_midpoint=BV1_bin)] = variable_to_be_binned.where(bin_index).mean()
        else:
            bin_mean_variable.loc[dict(BV1_bin_midpoint=BV1_bin)] = np.nan

    return bin_mean_variable, bin_number_of_samples


def bin_by_two_variables(variable_to_be_binned, BV1, BV2, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector,
                         lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector):
    """
    Bin input variable by two other variables (2D version of the function above). BV1 and BV2 are the variables
    to be used to create the bins
    :param variable_to_be_binned: variable to find the mean in each bin of BV1 an BV2 (e.g. precipitation)
    :param BV1: first variable used to create the bins (e.g. column saturation fraction)
    :param BV2: second variable used to create the bins
    :param lower_BV1_bin_limit_vector: lower bin limits of BV1
    :param upper_BV1_bin_limit_vector: upper bin limits of BV1
    :param lower_BV2_bin_limit_vector: lower bin limits of BV2
    :param upper_BV2_bin_limit_vector: upper bin limits of BV2
    :return bin_mean_variable, bin_number_pos_variable, bin_number_of_samples:
    """
    # Define bins
    BV1_bin_midpoint = (lower_BV1_bin_limit_vector + upper_BV1_bin_limit_vector) / 2

    lower_BV1_bin_limit_DA = xr.DataArray(lower_BV1_bin_limit_vector, coords=[BV1_bin_midpoint],
                                          dims=['BV1_bin_midpoint'])
    upper_BV1_bin_limit_DA = xr.DataArray(upper_BV1_bin_limit_vector, coords=[BV1_bin_midpoint],
                                          dims=['BV1_bin_midpoint'])
    number_of_BV1_bins = len(BV1_bin_midpoint)

    BV2_bin_midpoint = (lower_BV2_bin_limit_vector + upper_BV2_bin_limit_vector) / 2;

    lower_BV2_bin_limit_DA = xr.DataArray(lower_BV2_bin_limit_vector, coords=[BV2_bin_midpoint],
                                          dims=['BV2_bin_midpoint'])
    upper_BV2_bin_limit_DA = xr.DataArray(upper_BV2_bin_limit_vector, coords=[BV2_bin_midpoint],
                                          dims=['BV2_bin_midpoint'])
    number_of_BV2_bins = len(BV2_bin_midpoint);

    # Instantiate composite variable
    coords = {'BV2_bin_midpoint': BV2_bin_midpoint, 'BV1_bin_midpoint': BV1_bin_midpoint}
    dims = ['BV2_bin_midpoint', 'BV1_bin_midpoint']

    bin_number_of_samples = xr.DataArray(
        np.full((len(lower_BV2_bin_limit_vector), len(lower_BV1_bin_limit_DA)), np.nan), dims=dims, coords=coords)
    bin_mean_variable = bin_number_of_samples.copy()
    bin_number_pos_variable = bin_number_of_samples.copy()

    # Calculate bin mean and number of positive values in each bin
    for BV1_bin in BV1_bin_midpoint:
        for BV2_bin in BV2_bin_midpoint:
            bin_index = (BV1 >= lower_BV1_bin_limit_DA.where(lower_BV1_bin_limit_DA.BV1_bin_midpoint == BV1_bin,
                                                             drop=True).values) & (BV1 <= upper_BV1_bin_limit_DA.where(
                upper_BV1_bin_limit_DA.BV1_bin_midpoint == BV1_bin, drop=True).values) & (
                                    BV2 >= lower_BV2_bin_limit_DA.where(
                                lower_BV2_bin_limit_DA.BV2_bin_midpoint == BV2_bin, drop=True).values) & (
                                    BV2 <= upper_BV2_bin_limit_DA.where(
                                upper_BV2_bin_limit_DA.BV2_bin_midpoint == BV2_bin, drop=True).values)
            bin_number_of_samples.loc[dict(BV2_bin_midpoint=BV2_bin, BV1_bin_midpoint=BV1_bin)] = bin_index.sum()

            if np.isfinite(variable_to_be_binned.where(bin_index)).sum() > 0:
                bin_mean_variable.loc[
                    dict(BV2_bin_midpoint=BV2_bin, BV1_bin_midpoint=BV1_bin)] = variable_to_be_binned.where(
                    bin_index).mean()
                bin_number_pos_variable.loc[dict(BV2_bin_midpoint=BV2_bin, BV1_bin_midpoint=BV1_bin)] = (
                            variable_to_be_binned.where(bin_index) > 0).sum()
            else:
                bin_mean_variable.loc[dict(BV2_bin_midpoint=BV2_bin, BV1_bin_midpoint=BV1_bin)] = np.nan
                bin_number_pos_variable.loc[dict(BV2_bin_midpoint=BV2_bin, BV1_bin_midpoint=BV1_bin)] = np.nan

    return bin_mean_variable, bin_number_pos_variable, bin_number_of_samples


def calculate_csf_precipitation_binned_composites(csf, precipitation_rate, year, fname_datasets_for_simulation):
    """
    Main computational routine to bin precipitation and column saturation fraction. Save results as netcdf files.
    :param csf: column saturation fraction data array
    :param precipitation_rate: precipitation data array
    :param year: year to process
    :param fname_datasets_for_simulation: filename for output file
    :return: no return, results are saved to netcdf file
    """
    current_year_string = str(year)

    while len(current_year_string) < 4:  # make sure "year" of files has 4 digits for consistent file naming convention
        current_year_string = '0' + current_year_string

    ##############################################################################################
    ####  Calculate Backwards, Forwards and Center Differences of CSF and Precipitation Rate  ####
    ##############################################################################################
    print('Calculating Differences')
    delta_csf_leading, delta_csf_lagging, delta_csf_centered = calculate_backward_forward_center_difference(csf)
    delta_precipitation_rate_leading, delta_precipitation_rate_lagging, delta_precipitation_rate_centered = \
        calculate_backward_forward_center_difference(precipitation_rate)

    #########################################
    ####  Bin Precipitation Rate By CSF  ####
    #########################################
    print('Binning and Compositing')

    # Define bins #
    lower_BV1_bin_limit_vector = np.arange(0, 1., 0.025)  # BV1 is CSF in this case

    upper_BV1_bin_limit_vector = np.arange(0 + 0.025, 1 + 0.025, 0.025)

    bin_mean_precipitation_rate, bin_number_of_samples = bin_by_one_variable(precipitation_rate, csf,
                                                                             lower_BV1_bin_limit_vector,
                                                                             upper_BV1_bin_limit_vector)

    ####  Output Data as NetCDF  ####

    # Name variables #
    bin_number_of_samples.name = 'bin_number_of_samples'
    bin_mean_precipitation_rate.name = 'bin_mean_variable'

    # Add year dimension to all variables #
    bin_number_of_samples = bin_number_of_samples.assign_coords(year=year).expand_dims('year')
    bin_mean_precipitation_rate = bin_mean_precipitation_rate.assign_coords(year=year).expand_dims('year')

    # Merge all neccessary dataarrays to a single dataset #
    output_dataset = xr.merge([bin_number_of_samples, bin_mean_precipitation_rate])

    # Add desired attributes #
    output_dataset.attrs['Comments'] = \
        'Binning variables 1 (BV1) is column saturation fraction [Kg Kg^-1], Bin mean variable is ' \
        'precipitation rate in [mm day^-1]'

    # Output dataset to NetCDF #
    output_dataset.to_netcdf(
        fname_datasets_for_simulation + '_CSF_binned_precipitation_rate_' + current_year_string + '.nc', 'w')

    #########################################
    ####  Bin CSF By Precipitation Rate  ####
    #########################################
    print('Binning and Compositing')

    # Define bins #
    lower_BV1_bin_limit_vector = np.arange(0, 150., 1.)  # BV1 is precipitation rate in this case

    upper_BV1_bin_limit_vector = np.arange(0 + 1., 150 + 1., 1.0)

    bin_mean_csf, bin_number_of_samples = bin_by_one_variable(csf, precipitation_rate, lower_BV1_bin_limit_vector,
                                                              upper_BV1_bin_limit_vector)

    ####  Output Data as NetCDF  ####

    # Name variables #
    bin_number_of_samples.name = 'bin_number_of_samples'
    bin_mean_csf.name = 'bin_mean_variable'

    # Add year dimension to all variables #
    bin_number_of_samples = bin_number_of_samples.assign_coords(year=year).expand_dims('year')
    bin_mean_csf = bin_mean_csf.assign_coords(year=year).expand_dims('year')

    # Merge all neccessary dataarrays to a single dataset #
    output_dataset = xr.merge([bin_number_of_samples, bin_mean_csf])

    # Add desired attributes #
    output_dataset.attrs['Comments'] = \
        'Binning variables 1 (BV1) is precipitation rate in [mm day^-1], Bin mean variable is ' \
        'column saturation fraction [Kg Kg^-1]'

    # Output dataset to NetCDF #
    output_dataset.to_netcdf(
        fname_datasets_for_simulation + '_precipitation_rate_binned_CSF_' + current_year_string + '.nc', 'w')

    #################################################
    ####  Bin By Both Precipitation Rate and CSF ####
    #################################################
    print('Binning and Compositing')

    # Define bins #
    lower_BV1_bin_limit_vector = np.arange(0.0, 0.95 + 0.05, 0.05)  # CSF

    upper_BV1_bin_limit_vector = np.arange(0.05, 1 + 0.05, 0.05)  # CSF

    lower_BV2_bin_limit_vector = np.concatenate(
        [np.arange(0., 5. + 5., 5.), np.arange(10., 90. + 10., 10.)])  # Precipitation rate

    upper_BV2_bin_limit_vector = np.concatenate(
        [np.arange(5., 10. + 5., 5.), np.arange(20., 100. + 10., 10.)])  # Precipitation rate

    bin_mean_precipitation_rate, bin_number_pos_precipitation_rate, bin_number_of_samples_precipitation_rate = \
        bin_by_two_variables(precipitation_rate, csf, precipitation_rate, lower_BV1_bin_limit_vector,
                             upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)

    bin_mean_delta_csf_leading, bin_number_pos_delta_csf_leading, bin_number_of_samples_leading = bin_by_two_variables(
        delta_csf_leading, csf, precipitation_rate, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector,
        lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)
    bin_mean_delta_precipitation_rate_leading, bin_number_pos_delta_precipitation_rate_leading, _ = bin_by_two_variables(
        delta_precipitation_rate_leading, csf, precipitation_rate, lower_BV1_bin_limit_vector,
        upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)

    bin_mean_delta_csf_lagging, bin_number_pos_delta_csf_lagging, bin_number_of_samples_lagging = bin_by_two_variables(
        delta_csf_lagging, csf, precipitation_rate, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector,
        lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)
    bin_mean_delta_precipitation_rate_lagging, bin_number_pos_delta_precipitation_rate_lagging, _ = bin_by_two_variables(
        delta_precipitation_rate_lagging, csf, precipitation_rate, lower_BV1_bin_limit_vector,
        upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)

    bin_mean_delta_csf_centered, bin_number_pos_delta_csf_centered, bin_number_of_samples_centered = bin_by_two_variables(
        delta_csf_centered, csf, precipitation_rate, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector,
        lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)
    bin_mean_delta_precipitation_rate_centered, bin_number_pos_delta_precipitation_rate_centered, _ = bin_by_two_variables(
        delta_precipitation_rate_centered, csf, precipitation_rate, lower_BV1_bin_limit_vector,
        upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)

    ####  Output Data as NetCDF  ####

    # Name variables #
    bin_number_of_samples_precipitation_rate.name = 'bin_number_of_samples_precipitation_rate'
    bin_number_of_samples_leading.name = 'bin_number_of_samples_leading'
    bin_number_of_samples_lagging.name = 'bin_number_of_samples_lagging'
    bin_number_of_samples_centered.name = 'bin_number_of_samples_centered'

    bin_mean_precipitation_rate.name = 'bin_mean_precipitation_rate'

    bin_mean_delta_csf_leading.name = 'bin_mean_delta_csf_leading'
    bin_mean_delta_precipitation_rate_leading.name = 'bin_mean_delta_precipitation_rate_leading'
    bin_number_pos_delta_csf_leading.name = 'bin_number_pos_delta_csf_leading'
    bin_number_pos_delta_precipitation_rate_leading.name = 'bin_number_pos_delta_precipitation_rate_leading'

    bin_mean_delta_csf_lagging.name = 'bin_mean_delta_csf_lagging'
    bin_mean_delta_precipitation_rate_lagging.name = 'bin_mean_delta_precipitation_rate_lagging'
    bin_number_pos_delta_csf_lagging.name = 'bin_number_pos_delta_csf_lagging'
    bin_number_pos_delta_precipitation_rate_lagging.name = 'bin_number_pos_delta_precipitation_rate_lagging'

    bin_mean_delta_csf_centered.name = 'bin_mean_delta_csf_centered'
    bin_mean_delta_precipitation_rate_centered.name = 'bin_mean_delta_precipitation_rate_centered'
    bin_number_pos_delta_csf_centered.name = 'bin_number_pos_delta_csf_centered'
    bin_number_pos_delta_precipitation_rate_centered.name = 'bin_number_pos_delta_precipitation_rate_centered'

    # Add year dimension to all variables #
    bin_number_of_samples_precipitation_rate = bin_number_of_samples_precipitation_rate.assign_coords(
        year=year).expand_dims('year')
    bin_number_of_samples_leading = bin_number_of_samples_leading.assign_coords(year=year).expand_dims('year')
    bin_number_of_samples_lagging = bin_number_of_samples_lagging.assign_coords(year=year).expand_dims('year')
    bin_number_of_samples_centered = bin_number_of_samples_centered.assign_coords(year=year).expand_dims('year')

    bin_mean_precipitation_rate = bin_mean_precipitation_rate.assign_coords(year=year).expand_dims('year')

    bin_mean_delta_csf_leading = bin_mean_delta_csf_leading.assign_coords(year=year).expand_dims('year')
    bin_mean_delta_precipitation_rate_leading = bin_mean_delta_precipitation_rate_leading.assign_coords(
        year=year).expand_dims('year')
    bin_number_pos_delta_csf_leading = bin_number_pos_delta_csf_leading.assign_coords(year=year).expand_dims('year')
    bin_number_pos_delta_precipitation_rate_leading = bin_number_pos_delta_precipitation_rate_leading.assign_coords(
        year=year).expand_dims('year')

    bin_mean_delta_csf_lagging = bin_mean_delta_csf_lagging.assign_coords(year=year).expand_dims('year')
    bin_mean_delta_precipitation_rate_lagging = bin_mean_delta_precipitation_rate_lagging.assign_coords(
        year=year).expand_dims('year')
    bin_number_pos_delta_csf_lagging = bin_number_pos_delta_csf_lagging.assign_coords(year=year).expand_dims('year')
    bin_number_pos_delta_precipitation_rate_lagging = bin_number_pos_delta_precipitation_rate_lagging.assign_coords(
        year=year).expand_dims('year')

    bin_mean_delta_csf_centered = bin_mean_delta_csf_centered.assign_coords(year=year).expand_dims('year')
    bin_mean_delta_precipitation_rate_centered = bin_mean_delta_precipitation_rate_centered.assign_coords(
        year=year).expand_dims('year')
    bin_number_pos_delta_csf_centered = bin_number_pos_delta_csf_centered.assign_coords(year=year).expand_dims('year')
    bin_number_pos_delta_precipitation_rate_centered = bin_number_pos_delta_precipitation_rate_centered.assign_coords(
        year=year).expand_dims('year')

    # Merge all neccessary data arrays to a single dataset #
    output_dataset = xr.merge(
        [bin_number_of_samples_precipitation_rate, bin_number_of_samples_leading, bin_number_of_samples_lagging,
         bin_number_of_samples_centered, bin_mean_precipitation_rate, bin_mean_delta_csf_leading,
         bin_mean_delta_precipitation_rate_leading, bin_number_pos_delta_csf_leading,
         bin_number_pos_delta_precipitation_rate_leading, bin_mean_delta_csf_lagging,
         bin_mean_delta_precipitation_rate_lagging, bin_number_pos_delta_csf_lagging,
         bin_number_pos_delta_precipitation_rate_lagging, bin_mean_delta_csf_centered,
         bin_mean_delta_precipitation_rate_centered, bin_number_pos_delta_csf_centered,
         bin_number_pos_delta_precipitation_rate_centered])

    # Add desired attributes #
    output_dataset.attrs['Comments'] = \
        'Binning variable 1 (BV1) is column saturation fraction [Kg Kg^-1], binning variable 2 (BV2) ' \
        'is precipitation rate [mm day^-1]'

    # Output dataset to NetCDF #
    output_dataset.to_netcdf(
        fname_datasets_for_simulation + '_CSF_precipitation_binned_data' + '_' + current_year_string + '.nc', 'w')


def process_binned_single_variable_dataset(filename):
    """
    Read single variable binned data from file, compute mean over all years in file.
    :param filename: filename for binned dataset
    :return: mean binned data
    """
    binned_dataset = xr.open_dataset(filename)

    # Calculate the bin means over all years #
    more_than_zero_obs_mask = binned_dataset.bin_number_of_samples.sum('year') > 0
    binned_dataset['bin_mean_variable'] = (binned_dataset.bin_mean_variable * binned_dataset.bin_number_of_samples).sum(
        'year').where(more_than_zero_obs_mask, other=np.nan) / binned_dataset.bin_number_of_samples.sum('year').where(
        more_than_zero_obs_mask, other=np.nan)

    # Sum number of observations in each bin over all years #
    binned_dataset['bin_number_of_samples'] = binned_dataset.bin_number_of_samples.sum('year')

    # Remove year dimension
    binned_dataset = binned_dataset.squeeze()

    return binned_dataset


def process_multiyear_binned_single_variable_dataset(list_of_files):
    """
    Read single variable binned data from multiple files and  compute mean over all years and files.
    This is done using xr.open_mfdataset and requires dask.
    :param list_of_files: list of input filenames
    :return: mean binned data
    """
    binned_dataset = xr.open_mfdataset(list_of_files, combine="by_coords")

    # Calculate the bin means over all years #
    more_than_zero_obs_mask = binned_dataset.bin_number_of_samples.sum('year') > 0
    binned_dataset['bin_mean_variable'] = (binned_dataset.bin_mean_variable * binned_dataset.bin_number_of_samples).sum(
        'year').where(more_than_zero_obs_mask, other=np.nan) / binned_dataset.bin_number_of_samples.sum('year').where(
        more_than_zero_obs_mask, other=np.nan)

    # Sum number of observations in each bin over all years #
    binned_dataset['bin_number_of_samples'] = binned_dataset.bin_number_of_samples.sum('year')

    # Remove year dimension
    binned_dataset = binned_dataset.squeeze()

    return binned_dataset


def process_binned_csf_precipitation_rate_dataset(filename):
    """
    Read data binned by two variables (e.g. precipitation and csf) and compute mean across all years.
    :param filename: input filename containing binned data
    :return binned_csf_precipitation_rate_dataset: dataset containing mean binned data
    """
    # binned_csf_precipitation_rate_dataset = xr.open_mfdataset(list_of_files, combine="by_coords")
    binned_csf_precipitation_rate_dataset = xr.open_dataset(filename)

    # Calculate the bin means over all years #
    more_than_zero_obs_mask_precipitation_rate = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_precipitation_rate.sum('year') > 0

    more_than_zero_obs_mask_leading = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_leading.sum('year') > 0

    more_than_zero_obs_mask_lagging = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_lagging.sum('year') > 0

    more_than_zero_obs_mask_centered = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_centered.sum('year') > 0

    binned_csf_precipitation_rate_dataset['bin_mean_precipitation_rate'] = \
        (binned_csf_precipitation_rate_dataset.bin_mean_precipitation_rate *
         binned_csf_precipitation_rate_dataset.bin_number_of_samples_precipitation_rate
         ).sum('year').where(more_than_zero_obs_mask_precipitation_rate, other=np.nan) / \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_precipitation_rate.sum(
            'year').where(more_than_zero_obs_mask_precipitation_rate, other=np.nan)

    binned_csf_precipitation_rate_dataset['bin_mean_delta_csf_leading'] = \
        (binned_csf_precipitation_rate_dataset.bin_mean_delta_csf_leading *
         binned_csf_precipitation_rate_dataset.bin_number_of_samples_leading
         ).sum('year').where(more_than_zero_obs_mask_leading, other=np.nan) / \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_leading.sum(
        'year').where(more_than_zero_obs_mask_leading, other=np.nan)

    binned_csf_precipitation_rate_dataset['bin_mean_delta_precipitation_rate_leading'] = \
        (binned_csf_precipitation_rate_dataset.bin_mean_delta_precipitation_rate_leading *
         binned_csf_precipitation_rate_dataset.bin_number_of_samples_leading
         ).sum('year').where(more_than_zero_obs_mask_leading, other=np.nan) / \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_leading.sum(
        'year').where(more_than_zero_obs_mask_leading, other=np.nan)

    binned_csf_precipitation_rate_dataset['bin_mean_delta_csf_lagging'] = \
        (binned_csf_precipitation_rate_dataset.bin_mean_delta_csf_lagging *
         binned_csf_precipitation_rate_dataset.bin_number_of_samples_lagging
         ).sum('year').where(more_than_zero_obs_mask_lagging, other=np.nan) / \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_lagging.sum(
        'year').where(more_than_zero_obs_mask_lagging, other=np.nan)

    binned_csf_precipitation_rate_dataset['bin_mean_delta_precipitation_rate_lagging'] = \
        (binned_csf_precipitation_rate_dataset.bin_mean_delta_precipitation_rate_lagging *
         binned_csf_precipitation_rate_dataset.bin_number_of_samples_lagging).sum(
        'year').where(more_than_zero_obs_mask_lagging, other=np.nan) / \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_lagging.sum(
        'year').where(more_than_zero_obs_mask_lagging, other=np.nan)

    binned_csf_precipitation_rate_dataset['bin_mean_delta_csf_centered'] = \
        (binned_csf_precipitation_rate_dataset.bin_mean_delta_csf_centered *
         binned_csf_precipitation_rate_dataset.bin_number_of_samples_centered).sum(
        'year').where(more_than_zero_obs_mask_centered, other=np.nan) / \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_centered.sum(
        'year').where(more_than_zero_obs_mask_centered, other=np.nan)

    binned_csf_precipitation_rate_dataset['bin_mean_delta_precipitation_rate_centered'] = \
        (binned_csf_precipitation_rate_dataset.bin_mean_delta_precipitation_rate_centered *
         binned_csf_precipitation_rate_dataset.bin_number_of_samples_centered).sum(
        'year').where(more_than_zero_obs_mask_centered, other=np.nan) / \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_centered.sum(
        'year').where(more_than_zero_obs_mask_centered, other=np.nan)

    # Sum number of observations in each bin over all years #
    binned_csf_precipitation_rate_dataset['bin_number_of_samples_precipitation_rate'] = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_precipitation_rate.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_of_samples_leading'] = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_leading.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_of_samples_lagging'] = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_lagging.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_of_samples_centered'] = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_centered.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_pos_delta_csf_leading'] = \
        binned_csf_precipitation_rate_dataset.bin_number_pos_delta_csf_leading.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_pos_delta_precipitation_rate_leading'] = \
        binned_csf_precipitation_rate_dataset.bin_number_pos_delta_precipitation_rate_leading.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_pos_delta_csf_lagging'] = \
        binned_csf_precipitation_rate_dataset.bin_number_pos_delta_csf_lagging.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_pos_delta_precipitation_rate_lagging'] = \
        binned_csf_precipitation_rate_dataset.bin_number_pos_delta_precipitation_rate_lagging.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_pos_delta_csf_centered'] = \
        binned_csf_precipitation_rate_dataset.bin_number_pos_delta_csf_centered.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_pos_delta_precipitation_rate_centered'] = \
        binned_csf_precipitation_rate_dataset.bin_number_pos_delta_precipitation_rate_centered.sum('year')

    # Remove year dimension
    binned_csf_precipitation_rate_dataset = binned_csf_precipitation_rate_dataset.squeeze()

    return binned_csf_precipitation_rate_dataset


def process_multiyear_binned_csf_precipitation_rate_dataset(list_of_files):
    """
    Read multiple files with data binned by two variables (e.g. precipitation and csf) and
    compute mean across all years. This uses xr.open_mfdataset and requires dask.
    :param list_of_files: list of input filenames containing binned data
    :return binned_csf_precipitation_rate_dataset: dataset containing mean binned data
    """
    binned_csf_precipitation_rate_dataset = xr.open_mfdataset(list_of_files, combine="by_coords")

    # Calculate the bin means over all years #
    more_than_zero_obs_mask_precipitation_rate = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_precipitation_rate.sum('year') > 0

    more_than_zero_obs_mask_leading = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_leading.sum('year') > 0

    more_than_zero_obs_mask_lagging = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_lagging.sum('year') > 0

    more_than_zero_obs_mask_centered = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_centered.sum('year') > 0

    binned_csf_precipitation_rate_dataset['bin_mean_precipitation_rate'] = \
        (binned_csf_precipitation_rate_dataset.bin_mean_precipitation_rate *
         binned_csf_precipitation_rate_dataset.bin_number_of_samples_precipitation_rate
         ).sum('year').where(more_than_zero_obs_mask_precipitation_rate, other=np.nan) / \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_precipitation_rate.sum(
            'year').where(more_than_zero_obs_mask_precipitation_rate, other=np.nan)

    binned_csf_precipitation_rate_dataset['bin_mean_delta_csf_leading'] = \
        (binned_csf_precipitation_rate_dataset.bin_mean_delta_csf_leading *
         binned_csf_precipitation_rate_dataset.bin_number_of_samples_leading
         ).sum('year').where(more_than_zero_obs_mask_leading, other=np.nan) / \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_leading.sum(
            'year').where(more_than_zero_obs_mask_leading, other=np.nan)

    binned_csf_precipitation_rate_dataset['bin_mean_delta_precipitation_rate_leading'] = \
        (binned_csf_precipitation_rate_dataset.bin_mean_delta_precipitation_rate_leading *
         binned_csf_precipitation_rate_dataset.bin_number_of_samples_leading
         ).sum('year').where(more_than_zero_obs_mask_leading, other=np.nan) / \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_leading.sum(
            'year').where(more_than_zero_obs_mask_leading, other=np.nan)

    binned_csf_precipitation_rate_dataset['bin_mean_delta_csf_lagging'] = \
        (binned_csf_precipitation_rate_dataset.bin_mean_delta_csf_lagging *
         binned_csf_precipitation_rate_dataset.bin_number_of_samples_lagging
         ).sum('year').where(more_than_zero_obs_mask_lagging, other=np.nan) / \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_lagging.sum(
            'year').where(more_than_zero_obs_mask_lagging, other=np.nan)

    binned_csf_precipitation_rate_dataset['bin_mean_delta_precipitation_rate_lagging'] = \
        (binned_csf_precipitation_rate_dataset.bin_mean_delta_precipitation_rate_lagging *
         binned_csf_precipitation_rate_dataset.bin_number_of_samples_lagging).sum(
            'year').where(more_than_zero_obs_mask_lagging, other=np.nan) / \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_lagging.sum(
            'year').where(more_than_zero_obs_mask_lagging, other=np.nan)

    binned_csf_precipitation_rate_dataset['bin_mean_delta_csf_centered'] = \
        (binned_csf_precipitation_rate_dataset.bin_mean_delta_csf_centered *
         binned_csf_precipitation_rate_dataset.bin_number_of_samples_centered).sum(
            'year').where(more_than_zero_obs_mask_centered, other=np.nan) / \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_centered.sum(
            'year').where(more_than_zero_obs_mask_centered, other=np.nan)

    binned_csf_precipitation_rate_dataset['bin_mean_delta_precipitation_rate_centered'] = \
        (binned_csf_precipitation_rate_dataset.bin_mean_delta_precipitation_rate_centered *
         binned_csf_precipitation_rate_dataset.bin_number_of_samples_centered).sum(
            'year').where(more_than_zero_obs_mask_centered, other=np.nan) / \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_centered.sum(
            'year').where(more_than_zero_obs_mask_centered, other=np.nan)

    # Sum number of observations in each bin over all years #
    binned_csf_precipitation_rate_dataset['bin_number_of_samples_precipitation_rate'] = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_precipitation_rate.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_of_samples_leading'] = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_leading.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_of_samples_lagging'] = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_lagging.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_of_samples_centered'] = \
        binned_csf_precipitation_rate_dataset.bin_number_of_samples_centered.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_pos_delta_csf_leading'] = \
        binned_csf_precipitation_rate_dataset.bin_number_pos_delta_csf_leading.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_pos_delta_precipitation_rate_leading'] = \
        binned_csf_precipitation_rate_dataset.bin_number_pos_delta_precipitation_rate_leading.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_pos_delta_csf_lagging'] = \
        binned_csf_precipitation_rate_dataset.bin_number_pos_delta_csf_lagging.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_pos_delta_precipitation_rate_lagging'] = \
        binned_csf_precipitation_rate_dataset.bin_number_pos_delta_precipitation_rate_lagging.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_pos_delta_csf_centered'] = \
        binned_csf_precipitation_rate_dataset.bin_number_pos_delta_csf_centered.sum('year')

    binned_csf_precipitation_rate_dataset['bin_number_pos_delta_precipitation_rate_centered'] = \
        binned_csf_precipitation_rate_dataset.bin_number_pos_delta_precipitation_rate_centered.sum('year')

    # Remove year dimension
    binned_csf_precipitation_rate_dataset = binned_csf_precipitation_rate_dataset.squeeze()

    return binned_csf_precipitation_rate_dataset


def output_B_L(filename, landfrac, mwa_ME_surface_to_850, mwa_ME_saturation_850_to_500,
               mwa_saturation_deficit_850_to_500):
    """
    Write B_L variables to file.
    :param filename: filename to write to.
    :param landfrac: landfraction array
    :param mwa_ME_surface_to_850: mass weighted average of moist enthalpy from the surface to 850hPa
    :param mwa_ME_saturation_850_to_500: mass weighted average of saturation moist enthalpy from 850hPa to 500hPa
    :param mwa_saturation_deficit_850_to_500: mass weighted average of saturation deficit from 850hPa to 500hPa
    :return:
    """
    # Name variables #
    landfrac.name = 'landfrac'
    mwa_ME_surface_to_850.name = 'mwa_ME_surface_to_850'
    mwa_ME_saturation_850_to_500.name = 'mwa_ME_saturation_850_to_500'
    mwa_saturation_deficit_850_to_500.name = 'mwa_saturation_deficit_850_to_500'

    # Add desired attributes #
    landfrac.attrs['Units'] = 'Fraction of land, 0 = all water, 1 = all land'
    mwa_ME_surface_to_850.attrs['Units'] = '[J Kg^-1]'
    mwa_ME_saturation_850_to_500.attrs['Units'] = '[J Kg^-1]'
    mwa_saturation_deficit_850_to_500.attrs['Units'] = '[Kg Kg^-1]'

    # Merge all neccessary dataarrays to a single dataset #
    output_dataset = xr.merge(
        [landfrac, mwa_ME_surface_to_850, mwa_ME_saturation_850_to_500, mwa_saturation_deficit_850_to_500])

    # Output dataset to NetCDF #
    output_dataset.to_netcdf(filename)