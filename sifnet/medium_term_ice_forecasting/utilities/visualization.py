"""
Contains tools to creates forecast visualizations and other related functions

Notes
-----


Examples
--------
"""

import os
import math
import shutil

import imageio
import numpy as np
from copy import copy
import matplotlib.pylab as plt
from datetime import timedelta
import matplotlib.patches as mpatches


def visualize_forecasts(forecasts, dates, initial_conditions, path,
                        truth=None, skip=True, mask=None, lat_bounds=None, lon_bounds=None):
    """
    Producing forecast visualizations

    :param forecasts: Array
        Forecasts. 
    :param dates: List of dates. 
        Should have one-to-one correspondence with forecasts.
    :param initial_conditions: Array.
        The sea ice conditions on the 'current' day of forecast.
    :param path: Str.
        Directory location for output.
    :param truth: Bool.
        Base truth. Same dimensions as forecasts. 
        Optional (in case this is a true forecast)
    :param skip: Bool.
        If true, visualizes only every 10th forecast. Otherwise visualizes 
        all forecasts. Default True.
    :param mask: np.ndarray: 2D.
        Used for landmasking.
    :param lat_bounds: optional. list/tuple of length 2. Defines the latitude bounds of the region.
        Used to provided latitude coordinates in maps.
        Should be of the form [Northern bound, Southern bound]
    :param lat_bounds: optional. list/tuple of length 2. Defines the longitude bounds of the region.
        Used to provided longitude coordinates in maps.
        Should be of the form [Eastern bound, Western bound]

    :return: None
    """

    # TODO: Allow user to add filename prefix
    # REVIEW: Convert matplotlib figure directly to numpy see https://matplotlib.org/3.1.1/gallery/misc/agg_buffer_to_array.html

    len_forecast_data = len(forecasts)
    len_forecast_dates = len(dates)

    if not len_forecast_data == len_forecast_dates:
        raise ValueError(
            "ERROR - Array length difference: Forecast array length: {} -- Forecast dates array length {}".format(len_forecast_data, len_forecast_dates))

    if not (truth is None):
        # Ensure forecast and truth data same length
        len_truth_data = len(truth)
        if not len_forecast_data == len_truth_data:
            raise ValueError(
                "ERROR - Array length difference: Forecast array length: {} -- Truth array length {}".format(len_forecast_data, len_truth_data))
    else:
        truth = np.zeros(shape=forecasts.shape)

    if lon_bounds is None or lat_bounds is None:
        print('WARNING: NO LAT/LON BOUNDS PROVIDED FOR FORECAST VISUALIZATIONS')
        adding_tics = False
    else:
        adding_tics = True

    if mask is not None:
        if type(mask) != np.ndarray:
            raise TypeError('mask must be a numpy array')
        if len(mask.shape) != 2:
            raise ValueError('mask must be 2D')
        mask1 = np.resize(mask, forecasts.shape)
        forecasts = np.ma.masked_where(mask1, forecasts)
        truth = np.ma.masked_where(mask1, truth)
        del mask1
        mask2 = np.resize(mask, initial_conditions.shape)
        initial_conditions = np.ma.masked_where(mask2, initial_conditions)
        del mask2

    num = len(forecasts)
    n_days = forecasts.shape[1]

    if path[-1] != '/':
        path = path + '/'

    # TODO: Create automatically in file system
    if not os.path.isdir("temp"):
        os.mkdir("temp")

    if adding_tics:  # add lat/long tics
        (y_tics, y_tic_labels), (x_tics, x_tic_labels) = calc_lat_lon_tics_labels(mask.shape, lat_bounds,
                                                                                  lon_bounds)

    # Generate GIF output
    for i in range(num):
        # Create GIF every n days
        if i % 10 == 0 or not skip:
            print(i)

            forecast = np.squeeze(forecasts[i])
            files = []

            for n in range(1, n_days+1):

                palette = copy(plt.cm.viridis)
                palette.set_bad('xkcd:grey')

                cmap = plt.cm.get_cmap('viridis', 2)
                cmap.set_bad('xkcd:grey')

                fig = plt.figure(figsize=[19.0, 10.0])
                plt.suptitle("Comparison between Observed and Forecasted Ice Presence"
                             "\n Over {} days from {}".format(n_days, dates[i]))

                # plt.subplots_adjust(left=0.125)
                plt.subplot(221)
                plt.title("Initial Observed Sea Ice Presence {}".format(dates[i]))
                plt.imshow(initial_conditions[i], cmap=cmap, vmin=0, vmax=1)
                if adding_tics:  # add lat/long tics
                    plt.xticks(x_tics, x_tic_labels)
                    plt.yticks(y_tics, y_tic_labels)

                plt.legend([mpatches.Patch(color=cmap(0)),
                            mpatches.Patch(color=cmap(1)),
                            mpatches.Patch(color='xkcd:grey')],
                           ['water',
                            'ice',
                            'land'])

                plt.subplot(222)
                plt.title("Observed Ice Presence on Day #{}: {}".format(n, dates[i] + timedelta(n)))
                plt.imshow(truth[i, n - 1, :, :],  cmap=palette)
                if adding_tics:  # add lat/long tics
                    plt.xticks(x_tics, x_tic_labels)
                    plt.yticks(y_tics, y_tic_labels)

                plt.legend([mpatches.Patch(color=cmap(0)),
                            mpatches.Patch(color=cmap(1)),
                            mpatches.Patch(color='xkcd:grey')],
                           ['water',
                            'ice',
                            'land'])
                plt.subplot(223)
                plt.title("Forecasted Ice Presence Probability on Day #{}: {}".format(n, dates[i] + timedelta(n)))
                plt.imshow(np.squeeze(forecast[n - 1, :, :]), cmap=palette, vmin=0, vmax=1)
                if adding_tics:  # add lat/long tics
                    plt.xticks(x_tics, x_tic_labels)
                    plt.yticks(y_tics, y_tic_labels)
                plt.colorbar()
                plt.subplot(224)
                plt.title("Forecasted Ice Presence on Day #{}: {}".format(n, dates[i] + timedelta(n)))
                plt.imshow(forecast[n - 1, :, :] > 0.5,  cmap=palette)
                if adding_tics:  # add lat/long tics
                    plt.xticks(x_tics, x_tic_labels)
                    plt.yticks(y_tics, y_tic_labels)

                plt.legend([mpatches.Patch(color=cmap(0)),
                            mpatches.Patch(color=cmap(1)),
                            mpatches.Patch(color='xkcd:grey')],
                           ['water',
                            'ice',
                            'land'])

                plt.savefig("temp/{}.png".format(n))
                files.append("temp/{}.png".format(n))
                plt.close()

            images = []

            for file in files:
                images.append(imageio.imread(file))

            # FIXME: Add more descriptive filename
            file = path + "Forecast_{}.gif".format(dates[i])
            imageio.mimwrite(file, images, fps=3)

    shutil.rmtree("temp")

def calc_lat_lon_tics_labels(shape, lats, lons):
    """
    Calculates the tic-locations and tic-labels given the raster shape and the desired lat and lon bounds
    :param shape: tuple/list of length 2 corresponding to shape of image
    :param lats: tuple/list of length 2 corresponding to latitude bounds of region
    :param lons: tuple/list of length2 correspoind to longitude bounds of region
    :return: ((y_tics, y_tic_labels), (x_tics, x_tic_labels))
    """
    if len(shape) != 2:
        raise TypeError('Received shape on length != 2')
    if len(lats) != 2:
        raise TypeError('Received lats of length != 2')
    if len(lons) != 2:
        raise TypeError('Received lons of length != 2')

    y = shape[0]
    x = shape[1]

    # number of tics will be at most (5 or 10 depending on y or x axis),
    # or at least the most such that each tic is separated by 10 pixels rounded up
    # in the end we actually will see (5 or 10) + 1 tics because we start from zero
    n_y_tics = min(math.ceil(y/10), 5)
    n_x_tics = min(math.ceil(x/10), 5)

    y_tics = np.arange(0, y+y/n_y_tics, y/n_y_tics)  # [0, y/n_y_tics, ... y]
    x_tics = np.arange(0, x+x/n_x_tics, x/n_x_tics)  # [0, x/n_x_tics, ... x]

    # edge cases
    if y_tics[-1] > y:
        y_tics = y_tics[:-1]

    if x_tics[-1] > x:
        x_tics = x_tics[:-1]

    y_tic_step = (lats[1] - lats[0])/n_y_tics
    x_tic_step = (lons[0] - lons[1])/n_x_tics

    y_tic_labels = np.arange(lats[0], lats[1]+y_tic_step, step=y_tic_step)
    if y_tic_labels[-1] < lats[1]:
        y_tic_labels = y_tic_labels[:-1]

    x_tic_labels = np.arange(lons[1], lons[0]+x_tic_step, step=x_tic_step)
    if x_tic_labels[-1] > lons[0]:
        x_tic_labels = x_tic_labels[:-1]

    y_tic_labels = [str(yt)[0:4] for yt in y_tic_labels]
    x_tic_labels = [str(xt)[0:5] for xt in x_tic_labels] # include extra digit for negative sign (Western Hemisphere)

    return (y_tics, y_tic_labels), (x_tics, x_tic_labels)