"""
Support for datasets

"""

import math
import numpy as np
from operator import xor

# from scipy.misc.pilutil import imresize
from netCDF4 import Dataset


def normalize_feature(feature, verbose=False):
    """
    Normalizes the given feature by subtracting by its mean and then dividing by the standard deviation.

    :param feature: array
    :param verbose: Bool. Print diagnostic information
    :return:
        normalized array of features
    """

    if verbose:
        print(locals())

    avg = np.mean(feature)
    std_dev = np.std(feature)
    feature_minus_avg = feature - avg
    feature = np.divide(feature_minus_avg, std_dev)

    if verbose:
        print("Normalized feature values: Feature {}, Avg {}, StdDev {}, Feature Minus Avg {}".format(feature, avg, std_dev,
                                                  feature_minus_avg))

    return feature, avg, std_dev


def calculate_degree_days(temp_data, dates, start_month, start_day, resolution, kind='FDD', normalize=True):
    """
    Calculate freezing, breakup or total degree days

    :param temp_data: array
        Temperature data in Kelvin.
    :param dates: str
        List of lists. Date format is strings [[yyyy, mm, dd]].
    :param start_month: int
        month of the date to reset to zero.
    :param start_day: int
        day of the date to reset to zero.
    :param resolution: list [int]
        raster resolution
    :param kind: enum
        FDD, MDD, or ADD
    :param normalize: bool
        Normalize data.
    :return: array
    """

    def freezing_filter(x):
        return min(0, x)

    def breakup_filter(x):
        return max(0, x)

    v_freezing_filter = np.vectorize(freezing_filter)
    v_breakup_filter = np.vectorize(breakup_filter)

    temp_data = temp_data - 273

    num_dates = len(dates)

    dd = np.ndarray(shape=(num_dates, resolution[0], resolution[1]))

    for i in range(num_dates):
        date = dates[i]

        if date.month == start_month and date.day == start_day:
            dd[i, :, :] = np.zeros(shape=(resolution[0], resolution[1]))
        else:
            if kind == "FDD":
                dd[i, :, :] = dd[i - 1, :, :] - v_freezing_filter(temp_data[i, :, :])
            elif kind == "MDD":
                dd[i, :, :] = dd[i - 1, :, :] + v_breakup_filter(temp_data[i, :, :])
            elif kind == "ADD":
                dd[i, :, :] = dd[i-1, :, :] - temp_data[i, :, :]
            else:
                raise Exception(kind + "is not a valid option. One of ADD, FDD, and MDD is required")

    if normalize:
        dd, _, _ = normalize_feature(dd)

    return dd


def calculate_lat_lon_indices(lats, lons, north, south, east, west):
    """
    Calculate latitude and longitude array indices based on user bounding box

    :param lats: Array.
        Latitude values.
    :param lons: Array.
        Longitude values.
    :param north: Float.
    :param south: Float.
    :param east: Float.
    :param west: Float.
        Must supply negative sign for Western hemisphere.
    :return: Tuple of lat/long bounding box indices.
    """

    lat0 = lats[0]
    delta_lat = lats[1] - lats[0]

    index_north = int(round((north - lat0) / delta_lat))
    index_south = int(round((south - lat0) / delta_lat))

    if index_north >= index_south:
        index_lat_low = math.floor(index_south)
        index_lat_high = math.floor(index_north)
    else:
        index_lat_low = math.floor(index_north)
        index_lat_high = math.ceil(index_south)

    lon0 = lons[0]
    delta_lon = lons[1] - lon0

    index_west = int(round((west - lon0) / delta_lon))
    index_east = int(round((east - lon0) / delta_lon))

    if index_east >= index_west:
        index_lon_low = math.floor(index_west)
        index_lon_high = math.ceil(index_east)
    else:
        index_lon_low = math.floor(index_east)
        index_lon_high = math.ceil(index_west)

    if not ((lats[index_lat_high] == south and lats[index_lat_low] == north) or (lats[index_lat_high] == north and lats[index_lat_low] == south)):
        raise Exception('Either south or north does not belong to list of possible latitudes')
    if not (east in lons and west in lons):
        if not((east + 180) in lons and (west + 180) in lons):
            raise Exception('Either east or west does not belong to list of possible longitudes')

    return index_lat_low, index_lat_high, index_lon_low, index_lon_high


def find_index_from_timeseries(times, desired_time):
    """
    Find index from time series for desired time

    :param times: Array.
    :param desired_time:
    :return: Integer.
        Index of single time from input data.

    Notes
    -----
    - Times should be sorted.
    - Returns the index i such that times[i] == desired time.
    - If no such i exists, then returns i so to minimize the difference (desired_time - times[i]).
    """

    first = times[0]
    hours_delta = times[1] - first

    # Calculate initial index where desired datetime is located within array
    data_index = int((desired_time - first) / hours_delta)

    # Initial index can not be bigger than size of times data
    len_times = len(times)
    if data_index > len_times:
        data_index = int(data_index / 2)

    if times[data_index] != desired_time:
        for x in range(0, len(times)):
            if times[x] == desired_time:
                data_index = x
                break

    # If there is NO time which is equal to desired time, find closest
    if desired_time not in times:
        # infinity(ish)
        min = 99999999999
        for x in range(0, len(times)):
            diff = abs(desired_time - times[x])
            if (diff < min):
                min = diff
                data_index = x

        print("Found closest time {} hours from desired.".format(min))

    return int(data_index)


def create_landmask(path, dates, roi, resolution, threshold=0.5, resize=True, bin=True,
                    verbose=False):
    """
    Create clipped landmask from ERA5 land-sea mask

    ERA5 netcdf file is utilized. For each date supplied,
    a corresponding landmask is generated.

    References:
    - https://apps.ecmwf.int/codes/grib/param-db?id=172

    :param path: Str. Full path to netcdf file.
    :param dates: List of all dates.
    :param roi: List of latitude and longitude values.
    :param resolution: List of single x and y dimension of raster cell.
    :param threhsold: Float. Valid values between 0-1.
        Determines values to include in mask.
    :param resize: Bool. Resize raster to resolution.
    :param bin: Bool. Report binary values or real fractions.
    :param verbose: Bool. Print diagnostic information.
    :return: 2d numpy array
        0-1 float range.
    """

    if verbose:
        # https://stackoverflow.com/questions/10724495/getting-all-arguments-and-values-passed-to-a-function/23982966
        print(locals())

    north, south, east, west = roi
    data = Dataset(path)

    # extract area of interest
    i_lat_low, i_lat_high, i_lon_low, i_lon_high = calculate_lat_lon_indices(
        np.array(data["latitude"]), np.array(data["longitude"]),
        north, south, east, west)

    if resolution is None:
        resize = False
        # eg. 380 x 600
        resolution = [i_lat_high - i_lat_low + 1,
                      i_lon_high - i_lon_low + 1]
    else:
        temp = [i_lat_high - i_lat_low + 1, i_lon_high - i_lon_low + 1]
        # in case the resolutions happen to be the same
        if resolution == temp:
            resize = False

    landmask = data['lsm'][0, i_lat_low:i_lat_high + 1, i_lon_low:i_lon_high + 1]

    if resize:
        landmask = imresize(landmask, (resolution[0], resolution[1]),
                            interp='lanczos', mode="F")

    # pad landmask with number of dates to ensure same array size when stacking
    landmask = np.resize(landmask, (len(dates), resolution[0],
                         resolution[1]))

    if (bin==False):
        return landmask
    # Threshold landmask
    # https://stackoverflow.com/questions/46214291/convert-numpy-array-to-0-or-1-based-on-threshold
    landmask = np.where(landmask[:, :, :] >= threshold, 1, 0).astype(int)

    return landmask