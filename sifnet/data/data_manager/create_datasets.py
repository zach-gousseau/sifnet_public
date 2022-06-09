"""
Create dataset

Central location to hold dataset creator functions.

Each dataset to be opened is encapsulated within a function.

Examples
--------

ice, sst = create_ostia_dataset(dates=[['2011','09', "01"],["2012","12","21"]],
                roi=[70, 51, -65, -95], save_individual_files=False,
                resolution=[50,100])

print(ice.shape)
>> (2,50,100)
print(sst.shape)
>> (2,50,100)

-----

data = create_erai_dataset(feature="sst", dates=[['2011','09', "01"],["2012","12","21"]],
            north=70, south=51, west=-95, east=-65, save_individual_files=False,
            resolution=[50,100])

print(data.shape)
>> (2,50,100)

"""

import os

import numpy as np

from netCDF4 import Dataset
# from scipy.misc.pilutil import imresize
import datetime

import sifnet.data.data_manager.dataset_functions as df


def create_ostia_dataset(dates, data_directory, roi, resolution):
    """
    Collects SST and sea ice concentration

    :param dates: List or Array
        datetime.date objects
    :param data_directory: str
        Path to OSTIA directory.
    :param roi: List of 4 floats. Latitude and longitude coordinates to form
        bounding box to extract data within. Format is [N, S, E, W] extents.
    :param resolution: float
        Tuple of output resolution i.e. [380, 600]. Default is None.

    :return: array
        Tuple of Numpy arrays (ice data, SST data). [n_dates, lat, lon] x2

    Notes
    -----
    - No normalization of data.
    - One sample for each time in dates is extracted.
    - Data will be sampled from within the region defined by North, South, East, West.
    - If a resolution is specified (eg. [380, 600]) then all samples will be resized to that resolution using nearest bilinear.
    """

    if not (type(dates) == list or type(dates) == np.ndarray):
        raise ValueError("Dates must be a list/array of dates")
    for d in dates:
        if type(d) != datetime.date:
            raise ValueError("Contents of dates must be datetime.dates")

    north, south, east, west = roi

    dates = np.array(dates)

    if not os.path.exists(data_directory):
        raise IOError("Path does not exist: {}".format(data_directory))

    # FIXME: Remove dependency on file being present
    tmp_file = os.path.join(data_directory,
                        "1985/19850101-UKMO-L4HRfnd-GLOB-v01-fv02-OSTIARAN.nc")

    if not os.path.exists(tmp_file):
        raise IOError("File does not exist: {}".format(tmp_file))

    # Load a temp file to calculate the latitude/longitude indices
    data = Dataset(tmp_file)

    i_lat_low, i_lat_high, i_lon_low, i_lon_high = df.calculate_lat_lon_indices(
        np.array(data["lat"]), np.array(data["lon"]),
        north, south, east, west)

    Resize = True

    if resolution is None:
        Resize = False
        # eg. 380 x 600
        resolution = [i_lat_high - i_lat_low + 1,
                      i_lon_high - i_lon_low + 1]
        print(resolution)
    else:
        temp = [i_lat_high - i_lat_low + 1, i_lon_high - i_lon_low + 1]
        # In case the resolutions happens to be the same
        if resolution == temp:
            Resize = False

    # Samples x Lon x Lat
    ice_fraction = np.ndarray(
        shape=(len(dates), resolution[0], resolution[1]))
    sst_data = np.ndarray(
        shape=(len(dates), resolution[0], resolution[1]))

    n = 0

    for date in dates:
        # Print date every 10 days
        if n % 10 == 0:
           print(date)

        year = date.year
        month = date.month
        day = date.day

        # Determine the filepath of this date
        if int(year) >= 2008:
            file = (data_directory + "/{}/{}{}{}120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc").format(
                year, year,
                str(month).zfill(2), str(day).zfill(2))
        else:
            file = (data_directory + "/{}/{}{}{}-UKMO-L4HRfnd-GLOB-v01-fv02-OSTIARAN.nc").format(
                year, year, str(month).zfill(2), str(day).zfill(2))

        data = Dataset(file)

        # In the below line, first index 0 corresponds to time dimension, which
        # only has one value
        ice_fraction_input = data["sea_ice_fraction"][0, i_lat_low:i_lat_high + 1,
                    i_lon_low:i_lon_high + 1].filled(0)

        sst_input = data["analysed_sst"][0, i_lat_low:i_lat_high + 1,
                    i_lon_low:i_lon_high + 1]

        # Convert to Celcius from Kelvin
        sst_input = sst_input - 273

        # Unmask the array. Missing values are now zero (if any exist)
        sst_input = sst_input.filled(0)

        # OSTIA has origin in bottom left, but should by convention be upper left
        ice_fraction_input = np.flipud(ice_fraction_input)
        sst_input = np.flipud(sst_input)

        np.nan_to_num(ice_fraction, copy=False)  # copy=False means changes are made in-place
        np.nan_to_num(sst_data, copy=False)

        if Resize:
            ice_fraction_input = imresize(ice_fraction_input,
                                          (resolution[0], resolution[1]),
                                          interp='bilinear', mode="F")

            sst_input = imresize(sst_input, (resolution[0], resolution[1]),
                                 interp='bilinear', mode="F")

        ice_fraction[n, :, :] = ice_fraction_input
        sst_data[n, :, :] = sst_input

        n = n + 1

    return ice_fraction, sst_data

def create_era_interim_dataset(dates, data_directory, roi, feature,
                               resolution=None):
    """
    Creates ERA-Interim dataset composed of one or more environmental features

    :param dates: str
        Dates in string format [[yyyy, mm, dd]] as list of lists.
    :param data_directory: str
        Path to ERA Interim directory.
    :param roi: List of 4 floats. Latitude and longitude coordinates to form
    bounding box to extract data within. Format is [N, S, E, W] extents.
    :param feature: String. Abbreviation of variable to extract from data
    i.e. "sst".
    :param resolution: float
        Tuple of output resolution i.e. [380, 600]. Default is None.

    :return: array
        Tuple of Numpy arrays (feature 1, feature n ).

    Notes
    -----
    - No normalization of data.
    - One sample for each time in dates is extracted.
    - Data will be sampled from within the region defined by North, South, East, West.
    - If a resolution is specified (eg. [380, 600]) then all samples will be resized to that resolution using bilinear.
    """

    valid_features = {"ci", "d2m", "e", "fg10", "msl", "mwd", "mwp", "sf",
                      "slhf", "smlt", "sshf", "ssr",
                      "sst", "str", "swh", "t2m", "tcc", "tp", "u10",
                      "v10"}

    if feature not in valid_features:
        raise ValueError(
            "Invalid feature. Feature must be one of:" + str(valid_features)
            + " but instead received {}".format(feature))

    if not os.path.exists(data_directory):
        raise IOError("Path does not exist: {}".format(data_directory))

    if not (type(dates) == list or type(dates) == np.ndarray):
        raise ValueError("Dates must be a list/array of dates")
    for d in dates:
        if type(d) != datetime.date:
            raise ValueError("Contents of dates must be datetime.dates")

    north, south, east, west = roi

    try:
        dates = np.array(dates)
    except Exception as e:
        print(e)

    # Load a temp file to calculate the latitude/longitude indices
    try:
        # FIXME: Remove reliance on single file for dimensions
        file = data_directory + "/1979-01/ERAI_ci_197901.nc"
        data = Dataset(file)
    except Exception as e:
        print(e)

    i_lat_low, i_lat_high, i_lon_low, i_lon_high = df.calculate_lat_lon_indices(
        np.array(data["latitude"]), np.array(data["longitude"]),
        north, south, east, west)

    Resize = True

    if resolution is None:
        Resize = False
        # eg. 380 x 600
        resolution = [i_lat_high - i_lat_low + 1, i_lon_high - i_lon_low + 1]
    else:
        temp = [i_lat_high - i_lat_low + 1, i_lon_high - i_lon_low + 1]
        # In case the resolutions happen to be the same
        if resolution == temp:
            Resize = False

    data = np.ndarray(shape=(len(dates), resolution[0], resolution[1]))

    n = 0

    # Open the first file
    year = dates[0].year
    month = dates[0].month
    # eg. 1981-09/ERAI_ci_1981_09.nc
    file = data_directory + "/{}-{}/ERAI_{}_{}{}.nc".format(year, str(month).zfill(2),
                                                            feature,
                                                            year, str(month).zfill(2))
    # TODO: Log information
    try:
        dataset = Dataset(file)
    except Exception as e:
        print(e)

    for date in dates:
        current_year = date.year
        current_month = date.month
        current_day = date.day

        if current_month != month or current_year != year:
            dataset.close()
            month = current_month
            year = current_year

            file = data_directory + "/{}-{}/ERAI_{}_{}{}.nc".format(year, str(month).zfill(2),
                                                                    feature,
                                                                    year,
                                                                    str(month).zfill(2))

            # TODO: Skip netCDF files with issue opening
            # TODO: Log information
            try:
                dataset = Dataset(file)
            except Exception as e:
                print(e)

        times = dataset['time']

        hours = (int(current_day) - 1) * 24

        # Noon (first sample is at 3am) so 3am + 9 = noon
        desired_time = hours + times[0] + 9

        # Index of desired time within timeseries
        i = df.find_index_from_timeseries(times, desired_time)

        # Extract feature data
        extract_data = dataset[feature][i, i_lat_low:i_lat_high + 1, i_lon_low:i_lon_high + 1]
        extract_data = extract_data.filled(0)


        if Resize:
            extract_data = imresize(extract_data, (resolution[0], resolution[1]),
                                 interp='bilinear', mode="F")

        # Fill in the corresponding location in the time series with this sample
        data[n, :, :] = extract_data
        n = n + 1

    return data

def create_era5_dataset(dates, data_directory, landmask, roi, feature, resolution=None):
    """
    Creates ERA5 dataset composed of one or more environmental features

    :param dates: str
        Dates in string format [[yyyy, mm, dd]] as list of lists.
    :param data_directory: str
        Path to ERA5 directory.
    :param landmask: arr
        landmask array
    :param roi: List of 4 floats. Latitude and longitude coordinates to form
    bounding box to extract data within. Format is [N, S, E, W] extents.
    :param feature: String. Abbreviation of variable to extract from data
    i.e. "sst".
    :param resolution: float
        Tuple of output resolution i.e. [380, 600]. Default is None.

    :return: array
        Numpy array.

    Notes
    -----
    - No normalization of data.
    - One sample for each time in dates is extracted.
    - Data will be sampled from within the region defined by North, South, East, West.
    - If a resolution is specified (eg. [380, 600]) then all samples will be resized to that resolution using bilinear.
    """

    # https://stackoverflow.com/questions/10724495/getting-all-arguments-and-values-passed-to-a-function/23982966
    #print(locals()); input()

    # NetCDF variable extraction is short name; file path contains long name
    valid_features = {"d2m": "2m_dewpoint_temperature",
                      "t2m": "2m_temperature",
                      "u10": "10m_u_component_of_wind",
                      "v10": "10m_v_component_of_wind",
                      "fg10": "10m_wind_gust_since_previous_post_processing",
                      "e": "evaporation", "msl": "mean_sea_level_pressure",
                      "mwd": "mean_wave_direction",
                      "mwp": "mean_wave_period", "siconc": "sea_ice_cover",
                      "sst": "sea_surface_temperature",
                      "swh": "significant_height_of_combined_wind_waves_and_swell",
                      "sf": "snowfall", "smlt": "snowmelt",
                      "slhf": "surface_latent_heat_flux",
                      "sshf": "surface_sensible_heat_flux",
                      "tcc": "total_cloud_cover",
                      "tp": "total_precipitation",
                      "ssrd": "surface_solar_radiation_downwards"}

    if feature not in valid_features.keys():
        raise ValueError(
            "Invalid feature. Feature must be one of:" + str(valid_features)
            + " but instead received {}".format(feature))
    else:
        feature_long_name = valid_features.get(feature)

    if not os.path.exists(data_directory):
        raise IOError("Path does not exist: {}".format(data_directory))

    if not (type(dates) == list or type(dates) == np.ndarray):
        raise ValueError("Dates must be a list/array of dates")
    for d in dates:
        if type(d) != datetime.date:
            raise ValueError("Contents of dates must be datetime.dates")

    north, south, east, west = roi

    try:
        dates = np.array(dates)
    except Exception as e:
        print(e)
    # Load a temp file to calculate the latitude/longitude indices
    try:
        # FIXME: Remove reliance on single file for dimensions
        file = data_directory + "/1979-01/ERA5_2m_temperature_197901.nc"
        data = Dataset(file)
    except Exception as e:
        print(e)

    i_lat_low, i_lat_high, i_lon_low, i_lon_high = df.calculate_lat_lon_indices(
        np.array(data["latitude"]), np.array(data["longitude"]),
        north, south, east, west)

    Resize = True

    if resolution is None:
        Resize = False
        # eg. 380 x 600
        resolution = [i_lat_high - i_lat_low + 1, i_lon_high - i_lon_low + 1]
    else:
        temp = [i_lat_high - i_lat_low + 1, i_lon_high - i_lon_low + 1]
        # In case the resolutions happen to be the same
        if resolution == temp:
            Resize = False

    data = np.ndarray(shape=(len(dates), resolution[0], resolution[1]))

    ind = 0
    if (feature == 'siconc') or (feature == 'sst'):
        # Load a temp file to calculate the ice concentration indices near land
        try:
            # FIXME: Remove reliance on single file for dimensions
            file = data_directory + "/1979-01/ERA5_sea_ice_cover_197901.nc"
            tmp = Dataset(file)
            # landmask_path = '/home/nazanin/workspace/NRC_repo/src/system/sifnet/medium_term_ice_forecasting/support_files/era5-landsea-mask.nc'
            # land_data = Dataset(landmask_path)
            extract_data = tmp['siconc'][0, i_lat_low:i_lat_high + 1, i_lon_low:i_lon_high + 1]
            tmp.close()
        except Exception as e:
            print(e)

        extract_data = extract_data.filled(0)

        if Resize:
            extract_data = imresize(extract_data, (resolution[0], resolution[1]),
                                 interp='bilinear', mode="F")
            landmask = imresize(landmask, (resolution[0], resolution[1]),
                                interp='lanczos', mode="F")
        ice = extract_data > 0.15   #ice-water thresholding

        # Determine if neighbor lies inside grid
        def valid_neighbour(array, i, j):
            if i > 0 and j > 0 and i < array.shape[0] and j < array.shape[1]:
                return True
            return False

        needs_fix = (((1-ice)*(1-landmask)) > 0.85)  #water points -- should be all ice this time of the year
        to_fix = {}
        for m in range(ice.shape[0]):
            for n in range(ice.shape[1]):
                if needs_fix[m, n]:
                    to_fix[(m,n)] = [[],[]]
                    sum = 0
                    cnt = 0
                    for (i, j) in [(m - 1, n - 1), (m - 1, n), (m - 1, n + 1), (m, n - 1), (m, n + 1), (m + 1, n - 1),
                                   (m + 1, n), (m + 1, n + 1)]:
                        if valid_neighbour(ice, i, j) and landmask[i, j] == 0 and ice[i,j] == 1:
                            to_fix[(m, n)][0].append(i)
                            to_fix[(m, n)][1].append(j)
                    if len(to_fix[(m,n)][0])<1:
                        del to_fix[(m,n)]
        needs_fix2 = ((ice*landmask) > 0.15) #ice and land

    # Open the first file
    year = dates[0].year
    month = dates[0].month
    # eg. 1981-09/ERA5_2m_temperature_1981_09.nc
    file = data_directory + "/{}-{}/ERA5_{}_{}{}.nc".format(year,
                                                          str(month).zfill(2),
                                                          feature_long_name, year,
                                                          str(month).zfill(2))
    print("Constructed ERA file path to read. ", file)

    try:
        dataset = Dataset(file)
    except Exception as e:
        print(e)

    for date in dates:
        current_year = date.year
        current_month = date.month
        current_day = date.day

        if current_month != month or current_year != year:
            dataset.close()
            month = current_month
            year = current_year

            file = data_directory + "/{}-{}/ERA5_{}_{}{}.nc".format(year, str(month).zfill(2),
                                                                  feature_long_name, year,
                                                                    str(month).zfill(2))

            try:
                dataset = Dataset(file)
            except Exception as e:
                print(e)

        times = dataset['time']

        hours = (int(current_day) - 1) * 24

        # Noon (first sample is at 3am) so 3am + 9 = noon
        desired_time = hours + times[0] + 9

        # Index of desired time within timeseries
        i = df.find_index_from_timeseries(times, desired_time)

        # Extract feature data
        try:
            extract_data = dataset[feature][i, i_lat_low:i_lat_high + 1, i_lon_low:i_lon_high + 1]

        except IndexError as e:
            print('INFO: Error encountered at date {}'.format(date))
            raise e
        extract_data = extract_data.filled(0)

        #print(extract_data.shape); print(date)

        if Resize:
            extract_data = imresize(extract_data, (resolution[0], resolution[1]),
                                 interp='bilinear', mode="F")

        if (feature == 'siconc') or (feature == 'sst'):
            for key in to_fix:
                extract_data[key] = np.mean(extract_data[to_fix[key]])
            # Fill in the corresponding location in the time series with this sample
            extract_data[needs_fix2] = 0
        data[ind, :, :] = extract_data
        ind = ind + 1
    return data