"""
Standard utility functions for computing and plotting performance of a trained model.
Matthew King, April 2019
"""
import os
import calendar
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from datetime import date, timedelta


import sifnet.medium_term_ice_forecasting.utilities.numpy_metrics as nm
from sifnet.medium_term_ice_forecasting.utilities.visualization import calc_lat_lon_tics_labels
from sifnet.medium_term_ice_forecasting.utilities.reliability_diagram_vsklearn import reliability_diagram



def standard_evaluate(ytrue, ypredict, mask=None, verbose=True):
    """
    Evaluate a model after a set of forecasts have been produced

    :param ytrue: Array
        The base truth. Must be same size as ypredict
    :param ypredict: Array
        The model's forecasts. Must be same size as ytrue.
    :param mask: Array
        Land mask.
    :param verbose: boolean
        Determines if the metrics should be printed to console.
    :return: triplet of Numpy Arrays
        (recall, precision, accuracy). Each on a per-day and average basis.
    """

    # TODO: Log results to file
    # FIXME: Add parameter for verbose

    if not ytrue.shape == ypredict.shape:
        raise ValueError("ytrue shape does not match ypredict shape. ytrue shape is: ", ytrue.shape,
                         " ypredict shape is: ", ypredict.shape)

    (m_recall, d_recall) = nm.np_recall(ytrue, ypredict, mask)
    (m_precision, d_precision) = nm.np_precision(ytrue, ypredict, mask)
    (m_acc, d_acc) = nm.np_accuracy(ytrue, ypredict, mask)

    if verbose:
        print("Accuracy: (mean, daily)")
        print(m_acc, d_acc)

        print("Recall: (mean, daily)")
        print(m_recall, d_recall)

        print("Precision: (mean, daily)")
        print(m_precision, d_precision)

    return (m_recall, d_recall), (m_precision, d_precision), (m_acc, d_acc)


def standard_performance_plot(val_metrics, test_metrics, path,
                              plot_title=None):
    """
    Produce plots of performance as a function of forecast day

    Take as input the output of standard evaluate (having been called once
    with validation data and once with test data).

    :param val_metrics: Array
        Metrics produced via standard_evaluate() on the validation set.
    :param test_metrics: Array
        Metrics produced via standard_evaluate() on the test set.
    :param path: Str
        Location to save figures.
    :param plot_title: Str
        Super title at top of the figure.
    :return: None
    """

    ((val_m_recall, val_d_recall), (val_m_precision, val_d_precision),
     (val_m_acc, val_d_acc)) = val_metrics
    ((test_m_recall, test_d_recall), (test_m_precision, test_d_precision),
     (test_m_acc, test_d_acc)) = test_metrics

    if plot_title is None:
        plot_title = "Metrics as a function of Forecast Day"

    fig = plt.figure(figsize=[19.0, 10.0])
    plt.suptitle(plot_title)

    # [0:5] takes the first few sig figs
    plt.subplot(231)
    plt.title("Val Precision. Mean: " + str(val_m_precision)[
                                        0:5])
    plt.plot(val_d_precision)

    plt.subplot(232)
    plt.title('Val Recall. Mean: ' + str(val_m_recall)[0:5])
    plt.plot(val_d_recall)

    plt.subplot(233)
    plt.title('Val Accuracy. Mean: ' + str(val_m_acc)[0:5])
    plt.plot(val_d_acc)

    plt.subplot(234)
    plt.title('Test Precision. Mean: ' + str(test_m_precision)[0:5])
    plt.plot(test_d_precision)

    plt.subplot(235)
    plt.title('Test Recall. Mean: ' + str(test_m_recall)[0:5])
    plt.plot(test_d_recall)

    plt.subplot(236)
    plt.title('Test Accuracy. Mean: ' + str(test_m_acc)[0:5])
    plt.plot(test_d_acc)

    term_recall = "Recall - Proportion of true positives which were correctly identified."
    term_precision = "Precision - Proportion of positive identifications which where actually correct."
    term_accuracy = "Accuracy - Overall fraction of correct predictions."

    plt.figtext(0.15, 0.01, term_recall, fontsize=12)
    plt.figtext(0.15, 0.025, term_precision, fontsize=12)
    plt.figtext(0.15, 0.04, term_accuracy, fontsize=12)

    plt.savefig(path)
    plt.close(fig=fig)


def standard_performance_plots(all_val_metrics, all_test_metrics, labels, path, plot_title=None):
    """
    Produce plots of performance as a function of forecast day

    Take as input the output of standard evaluate (having been called once
    with validation data and once with test data).

    :param all_val_metrics: List
        list of metrics produced via standard_evaluate() on the validation set.
    :param all_test_metrics: List
        list of metrics produced via standard_evaluate() on the test set.
    :param labels: List
        labels used to identify the graph
    :param path: Str
        Location to save figures.
    :param plot_title: Str
        Super title at top of the figure.
    :return: None
    """

    val_d_recalls = []
    val_d_precisions = []
    val_d_accs = []

    test_d_recalls = []
    test_d_precisions = []
    test_d_accs = []

    for i in range(len(all_val_metrics)):
        ((val_m_recall, val_d_recall), (val_m_precision, val_d_precision),
         (val_m_acc, val_d_acc)) = all_val_metrics[i]
        ((test_m_recall, test_d_recall), (test_m_precision, test_d_precision),
         (test_m_acc, test_d_acc)) = all_test_metrics[i]

        val_d_recalls.append(val_d_recall)
        val_d_precisions.append(val_d_precision)
        val_d_accs.append(val_d_acc)

        test_d_recalls.append(test_d_recall)
        test_d_precisions.append(test_d_precision)
        test_d_accs.append(test_d_acc)

    if plot_title is None:
        plot_title = "Metric Comparison"

    fig = plt.figure(figsize=[19.0, 10.0])
    plt.suptitle(plot_title)

    # [0:5] takes the first few sig figs
    plt.subplot(231)
    plt.title("Val Precision")
    for val_d_precision in val_d_precisions:
        plt.plot(val_d_precision)

    plt.legend(labels)

    plt.subplot(232)
    plt.title('Val Recall')
    for val_d_recall in val_d_recalls:
        plt.plot(val_d_recall)

    plt.legend(labels)

    plt.subplot(233)
    plt.title('Val Accuracy')
    for val_d_acc in val_d_accs:
        plt.plot(val_d_acc)

    plt.legend(labels)

    plt.subplot(234)
    plt.title('Test Precision')
    for test_d_precision in test_d_precisions:
        plt.plot(test_d_precision)

    plt.legend(labels)

    plt.subplot(235)
    plt.title('Test Recall')
    for test_d_recall in test_d_recalls:
        plt.plot(test_d_recall)

    plt.legend(labels)

    plt.subplot(236)
    plt.title('Test Accuracy')
    for test_d_acc in test_d_accs:
        plt.plot(test_d_acc)

    plt.legend(labels)

    term_recall = "Recall - Proportion of true positives which were correctly identified."
    term_precision = "Precision - Proportion of positive identifications which where actually correct."
    term_accuracy = "Accuracy - Overall fraction of correct predictions."

    plt.figtext(0.15, 0.01, term_recall, fontsize=12)
    plt.figtext(0.15, 0.025, term_precision, fontsize=12)
    plt.figtext(0.15, 0.04, term_accuracy, fontsize=12)

    plt.savefig(path)
    plt.close(fig=fig)


def single_dataset_standard_performance_plots(all_metrics, labels, path, plot_title=None, additional_text=None):
    """
    Produce plots of performance as a function of forecast day

    Take as input the output of standard evaluate (having been called once
    with validation data and once with test data).

    :param all_val_metrics: List
        list of metrics produced via standard_evaluate() on one dataset (validation, test, etc).
    :param labels: List
        labels used to identify the graph
    :param path: Str
        Location to save figures.
    :param plot_title: Str
        Super title at top of the figure.
    :param additional_text: Str
        Additional text to show at the bottom of the plot
    :return: None
    """

    daily_recalls = []
    daily_precisions = []
    daily_accs = []

    for i in range(len(all_metrics)):
        ((m_recall, d_recall), (m_precision, d_precision),
         (m_acc, d_acc)) = all_metrics[i]

        daily_recalls.append(d_recall)
        daily_precisions.append(d_precision)
        daily_accs.append(d_acc)

    N_days = len(d_recall)
    days = [i for i in range(1, N_days+1)]  # [1,2,3, .... 30]

    if plot_title is None:
        plot_title = "Metric Comparison"

    fig = plt.figure(figsize=[19.0, 10.0])
    fig.suptitle(plot_title)
    plt.subplots_adjust(top=0.85)
    ax1 = fig.add_subplot(131)
    ax1.set_title("Precision")
    ax1.set_xlabel('Forecast Lead Day')
    ax1.set_ylabel('Precision')
    for daily_precision in daily_precisions:
        ax1.plot(days, daily_precision)
     

    ax1.set_xlim(1, N_days)
#     ax1.set_ylim(0,1)
    ax1.grid()
    ax1.legend(labels, loc='best')

    ax2 = fig.add_subplot(132)
    ax2.set_title('Recall')
    ax2.set_xlabel('Forecast Lead Day')
    ax2.set_ylabel('Recall')
    for daily_recall in daily_recalls:
        ax2.plot(days, daily_recall)

    ax2.set_xlim(1, N_days)
#     ax2.set_ylim(0,1)
    ax2.grid()
    ax2.legend(labels, loc='best')

    ax3 = fig.add_subplot(133)
    plt.title('Accuracy')
    ax3.set_xlabel('Forecast Lead Day')
    ax3.set_ylabel('Accuracy')
    for daily_acc in daily_accs:
        plt.plot(days, daily_acc)

    ax3.set_xlim(1, N_days)
#     ax3.set_ylim(0.7,1.01)
    ax3.grid()
    ax3.legend(labels, loc='best')

    if additional_text is not None:
        plt.figtext(0.45, 0.05, additional_text)

    fig.savefig(path)
    plt.close(fig)


def metrics_to_yaml_format(se_metrics):
    """
    Merges the test array generated by each fold in cross validation.
    :param se_metrics: tuple
        triplet of Numpy Arrays returned by standard_evaluate
        (recall, precision, accuracy). Each on a per-day and average basis.

    """
    recall = se_metrics[0]
    mean_recall = float(recall[0])
    daily_recall = [float(x) for x in recall[1]]

    precision = se_metrics[1]
    mean_precision = float(precision[0])
    daily_precision = [float(x) for x in precision[1]]

    accuracy = se_metrics[2]
    mean_accuracy = float(accuracy[0])
    daily_accuracy = [float(x) for x in accuracy[1]]

    return dict(recall=dict(mean=mean_recall, daily=daily_recall),
                precision=dict(mean=mean_precision, daily=daily_precision),
                accuracy=dict(mean=mean_accuracy, daily=daily_accuracy))


def per_month_accuracy_plots(ytrue, ypredict, forecast_dates, mask, savepath, region_name,
                             climate_norm=None, persistence=None):
    """
     Produces a set of accuracy_per_day plots for the set of months which
     fit the criteria of having at least one forecast end within that month.
    :param ytrue: np.ndarray 4D [samples, days, lats, lons]
    :param ypredict: np.ndarray 4D [samples, days, lats, lons]
    :param forecast_dates: List of dates where len(dates) === samples
    :param mask: Landmask. np.ndarray 2D [lats, lons]
    :param savepath: Path to which figures shall be saved. Must already exist.
    :param region_name: String. Name of the region being evaluated.
    :param climate_norm: optional. np.ndarray 4D [samples, days, lats, lons]
    :param persistence: optional. np.ndarray 4D [samples, days, lats, lons]
    :return: None
    """
    if type(ytrue) != np.ndarray and type(ytrue) != np.memmap:
        raise TypeError('ytrue must be an ndarray or memmap')
    if len(ytrue.shape) != 4:
        raise ValueError('ytrue must be 4D')
    if type(ypredict) != np.ndarray and type(ypredict) != np.memmap:
        raise TypeError('ypredict must be an ndarray or memmap')
    if len(ypredict.shape) != 4:
        raise ValueError('ypredict must be 4D')
    if ytrue.shape != ypredict.shape:
        raise ValueError('ypredict and ytrue must have the same shape')
    if type(forecast_dates) != list:
        raise TypeError('forecast_dates must be a list')
    for i in range(1, len(forecast_dates)):
        if type(forecast_dates[i]) != date:
            raise TypeError('All elements of forecast_dates must be datetime.date objects')
        if forecast_dates[i] <= forecast_dates[i-1]:
            raise ValueError('Forecast dates must be strictly increasing')
    if len(forecast_dates) != len(ytrue):
        raise ValueError('forecast_dates, ytrue, ypredict must all have the same length (first dimension of ndarrays)')
    if type(mask) != np.ndarray:
        raise TypeError('mask bust be an ndarray')
    if len(mask.shape) != 2:
        raise ValueError('mask must be 2D')

    if not os.path.exists(savepath):
        raise IOError('savepath does not exist!')

    if climate_norm is not None:
        if type(climate_norm) != np.ndarray and type(climate_norm) != np.memmap:
            raise TypeError('climate norm must be an ndarray or memmap')
        if len(climate_norm.shape) != 4:
            raise ValueError('Climate norm must be 4D')
        if climate_norm.shape != ytrue.shape:
            raise ValueError('climate norm must be same shape as ytrue')
    else:
        print('WARNING: NO CLIMATE NORMAL DATA PROVIDED TO MONTHLY PLOTS')

    if persistence is not None:
        if type(persistence) != np.ndarray:
            raise TypeError('climate norm must be an ndarray')
        if len(persistence.shape) != 4:
            raise ValueError('Climate norm must be 4D')
        if persistence.shape != ytrue.shape:
            raise ValueError('climate norm must be same shape as ytrue')
    else:
        print('WARNING: NO PERSISTENCE DATA PROVIDED TO MONTHLY PLOTS')

    # final forecast day only
    endpoint = ytrue.shape[1]

    # finds each month which has at least one forecast ending in that month.
    # for each such month, returns a list of indices which may be used to quickly extract the relevant forecasts
    valid_unique_months, monthly_lookup_tables = unique_forecast_months_with_lookup_tables(forecast_dates, endpoint)

    for MM in valid_unique_months:

        print('At month {}'.format(MM))

        table = monthly_lookup_tables[MM]
        relevant_predictions = ypredict[table]  # extract the relevant samples
        relevant_truth = ytrue[table]
        month_long_name = calendar.month_name[MM]
        metrics = standard_evaluate(relevant_truth, relevant_predictions, mask, verbose=False)
        all_metrics = [metrics]
        labels = ['Model Predictions']

        if persistence is not None:
            relevant_Per = persistence[table]
            persistence_metrics = standard_evaluate(relevant_truth, relevant_Per, mask, verbose=False)
            all_metrics.append(persistence_metrics)
            labels.append('Persistence')

        if climate_norm is not None:
            relevant_cn = climate_norm[table]
            cn_metrics = standard_evaluate(relevant_truth, relevant_cn, mask, verbose=False)
            all_metrics.append(cn_metrics)
            labels.append('Climate Norm')

        plot_savepath = 'Month_{}_Forecast_plot.png'.format(month_long_name)
        plot_savepath = os.path.join(savepath, plot_savepath)

        first_forecast_date = forecast_dates[table[0]]
        fm = first_forecast_date.month
        fm = calendar.month_abbr[fm]
        fd = first_forecast_date.day
        last_forecast_date = forecast_dates[table[-1]]
        lm = last_forecast_date.month
        lm = calendar.month_abbr[lm]
        ld = last_forecast_date.day

        first_forecast_end_date = first_forecast_date + timedelta(endpoint)
        fm_e = first_forecast_end_date.month
        fm_e = calendar.month_abbr[fm_e]
        fd_e = first_forecast_end_date.day
        last_forecast_end_date = last_forecast_date + timedelta(endpoint)
        lm_e = last_forecast_end_date.month
        lm_e = calendar.month_abbr[lm_e]
        ld_e = last_forecast_end_date.day

        title = 'Metrics of Ice Presence Forecast \n ' \
                'Forecasts initiated from {}-{} to {}-{} \n' \
                'Endpoint from {}-{} to {}-{} with {} day lead time'\
                '\n' \
                '{}'.format(fm, fd, lm, ld, fm_e, fd_e, lm_e, ld_e, endpoint, region_name)
        plt.close()

        text = 'From N = {} individual forecasts'.format(len(table))
        single_dataset_standard_performance_plots(all_metrics, labels, plot_savepath, title, additional_text=text)

        #reliability diagrams
        title_reliability = '\n ' \
                'Forecasts initiated from {}-{} to {}-{} \n' \
                'Endpoint from {}-{} to {}-{} with {} day lead time'\
                '\n' \
                '{}'.format(fm, fd, lm, ld, fm_e, fd_e, lm_e, ld_e, endpoint, region_name)

        reliability_diagram(ytrue = relevant_truth, yconf = relevant_predictions, landmask = mask, num_bins= 10,
                            save_path = savepath, title = title_reliability, plot_strategy = 'uniform')

    print('Done monthly plots')


def per_month_accuracy_maps(ytrue, ypredict, climate_norm, forecast_dates, mask, savepath, region_name, atdays=None,
                            lat_bounds=None, lon_bounds=None):
    """
     Produces a set of accuracy_per_day plots and accuracy maps for the set of months which
     fit the criteria of having at least one forecast fall on the relevant forecast day within that month.
     If atday is not specified, the criteria is each month for which at least one forecast ends in that month.
    :param ytrue: np.ndarray 4D [samples, days, lats, lons]
    :param ypredict: np.ndarray 4D [samples, days, lats, lons]
    :param climate_norm: np.ndarray 4D [samples, days, lats, lons]
    :param forecast_dates: List of dates where len(dates) === samples
    :param mask: Landmask. np.ndarray 2D [lats, lons]
    :param savepath: Path to which figures shall be saved. Must already exist.
    :param region_name: String. Name of the region being evaluated.
    :param atdays: optional. List of integers specifying the forecast-days for which maps/plots will be produced.
                        By default every 15th day, including the final day, will be evaluated.
    :param lat_bounds: optional. Tuple or list of length 2 representing the latitude bounds of the region
    :param lon_bounds: optional. Tuple or list of length 2 representing the longitude bounds of the region

    :return: None
    """
    if type(ytrue) != np.ndarray and type(ytrue) != np.memmap:
        raise TypeError('ytrue must be an ndarray or memmap')
    if len(ytrue.shape) != 4:
        raise ValueError('ytrue must be 4D')
    if type(ypredict) != np.ndarray and type(ypredict) != np.memmap:
        raise TypeError('ypredict must be an ndarray or memmap')
    if len(ypredict.shape) != 4:
        raise ValueError('ypredict must be 4D')
    if ytrue.shape != ypredict.shape:
        raise ValueError('ypredict and ytrue must have the same shape')
    if type(forecast_dates) != list:
        raise TypeError('forecast_dates must be a list')
    for i in range(1, len(forecast_dates)):
        if type(forecast_dates[i]) != date:
            raise TypeError('All elements of forecast_dates must be datetime.date objects')
        if forecast_dates[i] <= forecast_dates[i-1]:
            raise ValueError('Forecast dates must be strictly increasing')
    if len(forecast_dates) != len(ytrue):
        raise ValueError('forecast_dates, ytrue, ypredict must all have the same length (first dimension of ndarrays)')
    if type(mask) != np.ndarray:
        raise TypeError('mask bust be an ndarray')
    if len(mask.shape) != 2:
        raise ValueError('mask must be 2D')

    if atdays is None:
        atdays = [d for d in range(15, ytrue.shape[1], 15)] + [ytrue.shape[1]]  # every 15th day up to and include final
    if type(atdays) != list:
        raise TypeError('atdays must be a list')
    for v in atdays:
        if type(v) != int:
            raise TypeError('all values of atdays must be int')
        if v <= 0 or v > ytrue.shape[1]:
            raise ValueError('all values of atdays must be 0<atdays<=forcast_duration')

    if not os.path.exists(savepath):
        raise IOError('savepath does not exist!')

    if type(climate_norm) != np.ndarray and type(climate_norm) != np.memmap:
        raise TypeError('climate norm must be an ndarray or memmap')
    if len(climate_norm.shape) != 4:
        raise ValueError('Climate norm must be 4D')
    if climate_norm.shape != ytrue.shape:
        raise ValueError('climate norm must be same shape as ytrue')

    if lon_bounds is None or lat_bounds is None:
        print('WARNING: NO LAT/LON BOUNDS PROVIDED FOR MAP')
        adding_tics = False
    else:
        adding_tics = True

    # iterate over forecast days for which maps will be producted
    for endpoint in atdays:

        print('At day {}'.format(endpoint))

        # finds each month which has at least one forecast ending in that month.
        # for each such month, returns a list of indices which may be used to quickly extract the relevant forecasts
        valid_unique_months, monthly_lookup_tables = unique_forecast_months_with_lookup_tables(forecast_dates, endpoint)

        for MM in valid_unique_months:

            print('At month {}'.format(MM))

            table = monthly_lookup_tables[MM]
            relevant_predictions = ypredict[table]
            relevant_climate_normal = climate_norm[table]
            relevant_truth = ytrue[table]
            month_long_name = calendar.month_name[MM]

            first_forecast_date = forecast_dates[table[0]]
            fm = first_forecast_date.month
            fm = calendar.month_abbr[fm]
            fd = first_forecast_date.day
            last_forecast_date = forecast_dates[table[-1]]
            lm = last_forecast_date.month
            lm = calendar.month_abbr[lm]
            ld = last_forecast_date.day

            first_forecast_end_date = first_forecast_date + timedelta(endpoint)
            fm_e = first_forecast_end_date.month
            fm_e = calendar.month_abbr[fm_e]
            fd_e = first_forecast_end_date.day
            last_forecast_end_date = last_forecast_date + timedelta(endpoint)
            lm_e = last_forecast_end_date.month
            lm_e = calendar.month_abbr[lm_e]
            ld_e = last_forecast_end_date.day

            # maps
            _, d_acc_maps = nm.np_accuracy_map(relevant_truth, relevant_predictions)
            relevant_acc_map = d_acc_maps[endpoint-1]
            _, d_n_acc_maps = nm.np_accuracy_map(relevant_truth, relevant_climate_normal)
            relevant_norm_acc_map = d_n_acc_maps[endpoint-1]

            palette = copy(plt.cm.viridis)
            palette.set_bad('xkcd:grey')

            # apply mask
            relevant_acc_map = np.ma.masked_where(mask, relevant_acc_map)
            relevant_norm_acc_map = np.ma.masked_where(mask, relevant_norm_acc_map)

            plt.close()
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[17, 6.0])  # ,
            fig.suptitle('Accuracy of Ice Presence Forecast after {} days \n'
                         'Forecasts initiated from {}-{} to {}-{} \n'
                         'Evaluated {} days later from {}-{} to {}-{} \n '
                         '\n'
                         '{}'.format(endpoint,
                                     fm, fd, lm, ld,
                                     endpoint, fm_e, fd_e, lm_e, ld_e,
                                     region_name))

            ax0, ax1, ax2 = axes.flatten()

            im0 = ax0.imshow(relevant_acc_map, cmap=palette, vmin=0.5, vmax=1.)
            ax0.set_title('Model Accuracy')
            im1 = ax1.imshow(relevant_norm_acc_map, cmap=palette, vmin=0.5, vmax=1.)
            ax1.set_title('Climate Normal Accuracy')
            im2 = ax2.imshow(relevant_acc_map-relevant_norm_acc_map, cmap=palette, vmin=-0.3, vmax=0.2)
#             im2 = ax2.imshow(relevant_acc_map - relevant_norm_acc_map, cmap=palette)  # dynamic range via min/max
            ax2.set_title('Model Accuracy \n Improvement over Climate Normal')

            fig.colorbar(im0, ax=ax0, fraction=0.03, pad=0.03)  # fraction controls height of color bar
            fig.colorbar(im1, ax=ax1, fraction=0.03, pad=0.03)  # pad controls distance from left side of plot
            fig.colorbar(im2, ax=ax2, fraction=0.03, pad=0.03)

            if adding_tics:  # add lat/long tics
                (y_tics, y_tic_labels), (x_tics, x_tic_labels) = calc_lat_lon_tics_labels(mask.shape, lat_bounds,
                                                                                          lon_bounds)
                for axis in [ax0, ax1, ax2]:
                    axis.set_yticks(y_tics)
                    axis.set_yticklabels(y_tic_labels)
                    axis.set_xticks(x_tics)
                    axis.set_xticklabels(x_tic_labels)

            plt.figtext(0.45, 0.15, 'From N = {} individual forecasts'.format(len(table)))

            map_save_path = 'Accuracy_maps_Month_{}_Forecast_Day_{}.png'.format(month_long_name, endpoint)
            map_save_path = os.path.join(savepath, map_save_path)

            plt.savefig(map_save_path)

    print('Done monthly maps')


def unique_forecast_months_with_lookup_tables(dates, duration):
    """
    :return months: list of months found
    :return monthly_lookup_tables: month to index
    """
    months = []
    monthly_lookup_tables = {}
    td = timedelta(duration)
    for i in range(len(dates)):
        d = dates[i]
        month = (d + td).month
        if month not in months:
            months.append(month)
            monthly_lookup_tables[month] = [i]
        else:
            monthly_lookup_tables[month].append(i)
    return months, monthly_lookup_tables