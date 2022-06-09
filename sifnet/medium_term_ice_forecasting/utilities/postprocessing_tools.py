import csv
import os
from datetime import date, timedelta

import numpy as np
import tensorflow as tf
import yaml
from numba import cuda
from pkg_resources import resource_filename
from tensorflow.python.keras.models import load_model
import seaborn as sns
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import calendar

from sifnet.medium_term_ice_forecasting.utilities.standardized_evaluations import unique_forecast_months_with_lookup_tables
from sifnet.medium_term_ice_forecasting.utilities.standardized_evaluations import standard_evaluate
import sifnet.medium_term_ice_forecasting.utilities.standardized_evaluations as se
from sifnet.data.DatasetManager import DatasetManager
from sifnet.data.GeneratorFunctions import historical_all_channels, future_multi_channel, \
    future_single_channel_thresholded
from sifnet.medium_term_ice_forecasting.ice_presence.ice_presence_evaluation import evaluation_procedure
from sifnet.medium_term_ice_forecasting.utilities.model_utilities import custom_objects
from sifnet.medium_term_ice_forecasting.utilities.standardized_evaluations import metrics_to_yaml_format
from sifnet.medium_term_ice_forecasting.utilities.visualization import visualize_forecasts
from sifnet.medium_term_ice_forecasting.utilities.numpy_metrics import np_brier
from sifnet.utilities.gpu import restrict_to_available_gpu

"""
Functions the user may want to call after training is complete. Includes:
    - function to calculate metrics from aggregated data cubes
    - function to aggregate data cubes generated from each fold in cross validation
    - function to aggregate csv date file from each fold in cross validation
    - function to produce gifs from specified date
    - function to evaluate a saved model in a specified directory
    - function to resume training if a model crashed
    - and a bunch of other helper functions
"""


def calculate_kfold_metrics(dsm, savepath, model_name):
    """
     Runs independent operations from python console

     :param dsm: DatasetManager
             Dataset manager object needed to create generator
     :param savepath: string
             Location to save generated files (Includes plot of modeled predictions
             versus climate normal and persistence)
     :param model_name: string
             Name of the model. Used to create title and filename for plot.
     :return: tuple
             metrics for modeled predictions, persistence, climate normal as calculated by
             standard_evaluate
     """
    test_data_path = os.path.join(savepath, "aggregated_test-data.npy")
    test_pred_path = os.path.join(savepath, "aggregated_test_preds.npy")
    test_persist_path = os.path.join(savepath, 'aggregated_test_persistence.npy')
    test_norm_path = os.path.join(savepath, 'aggregated_test_climate_normals.npy')

    # caclulate test predictions
    test_data = np.load(test_data_path)
    test_pred = np.load(test_pred_path)

    landmask = dsm.raw_data[0, :, :, dsm.find_landmask_channel()]
    test_samples = test_data.shape[0]
    test_metrics = se.standard_evaluate(test_data, test_pred, landmask)

    matching_dimensions = False
    if os.path.exists(test_persist_path) and os.path.exists(test_norm_path):
        test_persist = np.load(test_persist_path)
        test_norms = np.load(test_norm_path)

        matching_dimensions = test_norms.shape == test_persist.shape and test_norms.shape[0] == test_samples and \
                              test_norms.shape[-2:] == landmask.shape and test_norms.shape[
                                  1] == dsm.forecast_days_forward
    else:
        raise FileNotFoundError("{0} and {1} may not exist".format(test_persist_path, test_norm_path))

    if not matching_dimensions:
        raise ValueError("Dimension mismatch")

    test_persist_metrics = se.standard_evaluate(test_data, test_persist, landmask)
    test_normal_metrics = se.standard_evaluate(test_data, test_norms, landmask)

    plot_name = model_name + ".png"
    plot_path = os.path.join(savepath, plot_name)

    plot_text = "Comparison of Model Predictions ({}) to Climate Normal and Persistence".format(model_name)

    se.single_dataset_standard_performance_plots([test_metrics, test_persist_metrics, test_normal_metrics],
                                                 ['Modeled', 'Persistence', 'Climate Normal'],
                                                 plot_path, plot_text)

    local_summary = dict()

    local_summary['test_metrics'] = metrics_to_yaml_format(test_metrics)
    local_summary['test_persist_metrics'] = metrics_to_yaml_format(test_persist_metrics)
    local_summary['test_normal_metrics'] = metrics_to_yaml_format(test_normal_metrics)

    cross_val_summary = 'cross_val_summary.yaml'
    if cross_val_summary not in os.listdir(savepath):
        with open(os.path.join(savepath, cross_val_summary), 'w') as f:
            yaml.safe_dump(local_summary, f)

    return test_metrics, test_persist_metrics, test_normal_metrics


def create_unified_kfold_array(savepath, folds, filename, days_of_historic_input, forecast_days_forward, data_type):
    """
    Merges the test array generated by each fold in cross validation without using dataset manager

    :param savepath: str
            path to a directory containing models evaluated at each fold
            e.g /work/IcePresence/NWT_Freeze_v2/H3-F30/model_name
    :param folds: int
            number of folds used by cross validation model has
    :param filename: string
            name of npy file to be merged
            e.g. test_climate_normals.npy
    :param days_of_historic_input: int
            number of days used to observe
    :param forecast_days_forward: int
            number of days in forecast
    :param data_type: numpy dtype
            datatype the numpy array should be saved as
    :return: ndarray
            array containing aggregated data cube
    """
    test_records_path = os.path.join(savepath, "test_record.yaml")

    with open(test_records_path, 'r') as file:
        test_records = yaml.safe_load(file)
        file.close()

    year_to_cube = dict()
    years = []
    dim = np.array([0, 0, 0, 0])
    for i in range(folds):
        years.extend(test_records[i])
        test_pred_path = os.path.join(savepath, "fold_" + str(i), filename)
        test_date_path = os.path.join(savepath, "fold_" + str(i), "test_pred_dates.csv")

        predictions = np.load(test_pred_path)

        dim[0] += predictions.shape[0]
        dim[1:4] = predictions.shape[1:4]

        begin = test_records["yearly_start"]
        end = test_records["yearly_end"]
        wrap_around = bool(end[0] < begin[0] or (end[0] == begin[0] and end[1] < begin[1]))

        with open(test_date_path, 'r') as csv_file:
            # skip first line
            next(csv_file)
            csv_reader = csv.reader(csv_file, delimiter=',')

            index_of_first_day = 0
            starting_year_of_period = -1
            for row in csv_reader:
                current_year = int(row[0][:4])
                current_month = int(row[0][5:7])
                current_day = int(row[0][8:])

                current_date = date(current_year, current_month, current_day)
                # find beginning and end of year containing current_date
                if wrap_around and current_month >= begin[0]:
                    end_date = date(current_year + 1, end[0], end[1])
                else:
                    end_date = date(current_year, end[0], end[1])
                if wrap_around and current_month < begin[0]:
                    begin_date = date(current_year - 1, begin[0], begin[1])
                else:
                    begin_date = date(current_year, begin[0], begin[1])

                delta_forcasts = timedelta(forecast_days_forward + 1)
                delta_historic = timedelta(days_of_historic_input - 1)

                # found and record first day of year
                if current_date == begin_date + delta_historic:
                    index_of_first_day = int(row[1])
                    starting_year_of_period = int(row[0][:4])
                # found last day of year, slice data cube and save into dictionary
                elif current_date == end_date - delta_forcasts:
                    index_of_last_day = int(row[1])
                    year_to_cube[starting_year_of_period] = predictions[index_of_first_day:index_of_last_day + 1, :, :,
                                                            :]

    years.sort()
    aggregated_cube = np.ndarray(shape=(dim[0], dim[1], dim[2], dim[3]), dtype=data_type)

    prev = 0
    cur = 0
    # aggregate data cubes at each year together in chronological order
    for year in years:
        ytc = year_to_cube[year]
        cur += ytc.shape[0]
        aggregated_cube[prev:cur] = ytc
        prev = cur

    aggregated_filename = 'aggregated_' + filename[:-3] + "npy"
    aggregated_file_path = os.path.join(savepath, aggregated_filename)
    np.save(aggregated_file_path, aggregated_cube)
    return aggregated_cube


def aggregate_dates(savepath, folds):
    """
    Aggregates the dates saved from each fold together

    :param savepath: str
            path to a directory containing models evaluated at each fold
            e.g /work/IcePresence/NWT_Freeze_v2/H3-F30/model_name
    :param folds: int
            number of folds used by cross validation model has
    """
    all_dates = []
    for i in range(folds):
        test_pred_dates = os.path.join(savepath, "fold_" + str(i), "test_pred_dates.csv")
        with open(test_pred_dates, 'r') as f:
            reader = csv.reader(f)
            dates_and_index = map(tuple, reader)

            dates = [x[0] for x in dates_and_index]
            dates.pop(0)
            all_dates.extend([date(int(x[:4]), int(x[5:7]), int(x[8:])) for x in dates])

    all_dates.sort()
    date_strings = [str(d) for d in all_dates]
    aggregated_dates_path = os.path.join(savepath, "aggregated_test_pred_dates.csv")
    with open(aggregated_dates_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'numpy_join_index'])
        for i in range(len(date_strings)):
            d = date_strings[i]
            writer.writerow([d, i])


def extract_gridded_daily_time_series(data_cube, days_index):
    """
    Slices a numpy array of ice presence forecasts on a specific forecast day

    :param data_cube: ndarray
           numpy array of ice presence forecasts
           dimensions (samples, forecast_day_forward, x_resolution, y_resolution)
    :param days_index: int
           forecast day to slice the data cube at
    :return: ndarray
           dimensions (samples, x_resolution, y_resolution)
    """
    if days_index > data_cube.shape[1]:
        raise IndexError("Forecast range does not include the prediction specified")

    partial = data_cube[:, days_index, :, :]

    return partial


def find_indices_in_year(savepath, begin, end, forecast_days_forward, days_of_historic_input):
    """
    Find indices an array that corresponds to year

    :param savepath: string
           directory where the cross validation results are
    :param begin: [int, int]
           start of the year
    :param end: [int, int]
           end of the year
    :param forecast_days_forward: int
           number of days forecasted
    :param days_of_historic_input: int
           number of days observed by the model
    :return: dict
           dictionary of year to indices in array that corresponds to that year
    """
    indices_in_year = dict()
    wrap_around = (end[0] < begin[0] or end[0] == begin[0] and end[1] < begin[1])
    # grab list of dates from date csv saved in directory
    dates = get_dates_from_csv(savepath)
    for i in range(len(dates)):
        current_date = dates[i]
        current_year = current_date.year
        current_month = current_date.month

        if wrap_around and current_month >= begin[0]:
            end_date = date(current_year + 1, end[0], end[1])
        else:
            end_date = date(current_year, end[0], end[1])
        if wrap_around and current_month < begin[0]:
            begin_date = date(current_year - 1, begin[0], begin[1])
        else:
            begin_date = date(current_year, begin[0], begin[1])

        delta_forecasts = timedelta(forecast_days_forward + 1)
        delta_historic = timedelta(days_of_historic_input - 1)

        if begin_date + delta_historic <= current_date <= end_date - delta_forecasts:
            if begin_date.year in indices_in_year:
                indices_in_year[begin_date.year].append(i)
            else:
                indices_in_year[begin_date.year] = [i]

    return indices_in_year


def extract_test_records(savepath):
    """
    Extracts relevant information from test_record.yaml inside cross validation main directory

    :param savepath: string
           location of cross validation main directory
    :return: (int, list[int], list[int]
           (number of folds used by this cross validation,
    """
    names = os.listdir(savepath)
    num_folds = 0
    for name in names:
        if 'fold' in name:
            num_folds += 1

    with open(os.path.join(savepath, 'test_record.yaml'), 'r') as f:
        test_record = yaml.safe_load(f)
        f.close()

    begin = test_record.pop('yearly_start')
    end = test_record.pop('yearly_end')

    years = []
    for i in range(num_folds):
        years.extend(test_record[i])
    if num_folds == 0:
        years = test_record[0]
    years.sort()

    return num_folds, years, begin, end, test_record


def str_to_date(s):
    """
    Returns a Python date object from date string saved in csv file

    :param s: string
             date string
    :return: Date
             corresponding Python date object
    """
    current_year = int(s[:4])
    current_month = int(s[5:7])
    current_day = int(s[8:])

    return date(current_year, current_month, current_day)


def get_dates_from_csv(savepath):
    """
    Extracts dates from csv file

    :param savepath: string
           path to directory containing csv file or direct path to csv file
           If the path is a directory, it is checked to contain 'aggregated_test_pred_dates.csv',
           'test_pred_dates.csv', and 'val_pred_dates.csv' in that order. The first found will be used.
    :return: list[dates]
    """
    if not os.path.exists(savepath):
        raise IOError('The provided path: \' {} \'  does not exist'.format(savepath))

    if os.path.isdir(savepath):
        if os.path.exists(os.path.join(savepath, 'aggregated_test_pred_dates.csv')):
            load_path = os.path.join(savepath, 'aggregated_test_pred_dates.csv')
        elif os.path.exists(os.path.join(savepath, 'test_pred_dates.csv')):
            load_path = os.path.join(savepath, 'test_pred_dates.csv')
        elif os.path.exists(os.path.join(savepath, 'val_pred_dates.csv')):
            load_path = os.path.join(savepath, 'val_pred_dates.csv')
        else:
            raise IOError('Provided path: \' {} \' is a directory and does not contain aggregated_test_pred_dates.csv, '
                          'test_pred_dates.csv, or val_pred_dates.csv'.format(savepath))
    elif savepath.endswith('.csv'):
        load_path = savepath
    else:
        raise IOError('The provided path \' {} \' is neither a directory nor a file ending with \'.csv\'')

    all_dates = []
    with open(load_path, 'r') as csv_file:
        next(csv_file)
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            current_date = str_to_date(row[0])
            all_dates.append(current_date)

    return all_dates


def visualize_specific_dates(selected_dates, savepath, all_dates, forecasts, initial_conditions,
                             truth, landmask=None, lat_bounds=None, lon_bounds=None):
    """
    Produces GIF visualizations of forecasts initiated at the specified dates
    :param selected_dates: list of datetime.date objects
                These are the dates for which forecasts will be visualized
    :param savepath: str. Path to directory where GIFs will be saved.
    :param all_dates: list of datetime.date objects
                These are all the dates with an associaited forecast
    :param forecasts: numpy.ndarray 5D shape [samples, forecast_duration, lats, lons]
                The forecasts from which a subset will be visualized
    :param initial_conditions: numpy.ndarray 4D shape [samples, lats, lons]
                Inital sea ice conditions from the start of each forecast
    :param truth: numpy.ndarray 5D shape [samples, forecast_duration, lats, lons]
                The base truth. Same shape as forecasts.
    :param landmask: optional. numpy.ndarray 2D shape [lats, lons]
                landmask which will be applied to visualization
    :param lat_bounds: optional. list/tuple of length 2. Defines the latitude bounds of the region.
                Used to provided latitude coordinates in maps.
                Should be of the form [Northern bound, Southern bound]
    :param lat_bounds: optional. list/tuple of length 2. Defines the longitude bounds of the region.
            Used to provided longitude coordinates in maps.
            Should be of the form [Eastern bound, Western bound]
    :return: None
    """
    if not os.path.exists(savepath):
        raise IOError('The specified savepath \' {} \' does not exist'.format(savepath))
    if not os.path.isdir(savepath):
        raise IOError('The specified savepath \' {} \' is not a directory'.format(savepath))

    table = []
    for i in range(len(selected_dates)):
        d = selected_dates[i]
        for j in range(len(all_dates)):
            ad = all_dates[j]
            if d == ad:
                table.append(int(j))
                break  # break from inner loop

    assert [all_dates[i] for i in table] == selected_dates, 'One or more elements of selected_dates ' \
                                                            'were not present in all_dates.'
    forecasts = forecasts[table]
    initial_conditions = initial_conditions[table]
    truth = truth[table]

    visualize_forecasts(forecasts, selected_dates, initial_conditions, savepath, truth, skip=False,
                        mask=landmask, lat_bounds=lat_bounds, lon_bounds=lon_bounds)


def evaluate_directory(path, directory, visualize_forecasts='best', val_years=None, test_years=None,
                       save_best_model_predicitons=False):
    """
    Evaluates each model in the specified directory.
    Based on the value of 'visualize_forecasts', the best model, all the models, or none of the model may
    have their forecasts visualized.

    :param path: str
            path to a directory containing some number of models
            e.g /work/IcePresence/NWT_Freeze_v2/H3-F30/
    :param visualize_forecasts: str
            string describing the desired forecast visualization behaviour.
            Valid options are 'all', 'none', 'best'
            'all' visualizes all models' forecasts.
            'none' does not visualize any forecasts.
            'best' visualizes forecasts of the model with highest val accuracy only.
    :param test_years: list of test years
            default [1990, 2003, 2012, 2013, 2017]
    :param val_years: list of val years
            default [1989, 2000, 2002, 2014, 2016]
    """

    if type(path) != str:
        raise TypeError('Received path of non-string type.')

    if not os.path.exists(path):
        raise IOError('Path does not exist.')

    if not os.path.isdir(path):
        raise ValueError('Path is not a directory.')

    if not path.endswith(os.path.sep):  # os.path.sep == '/' if on linux
        path = path + os.path.sep

    split_path = path.split(os.path.sep)
    range_descriptor = split_path[-2]  # e.g 'H3-F30'
    if len(range_descriptor.split('-')) != 2:
        raise ValueError('Last directory in path is not Hn1-Fn2 describing historic/forecast days')

    din, dout = range_descriptor.split('-')

    if not (din[1:].isdigit() and dout[1:].isdigit()):
        raise ValueError('Last directory in path is not Hn1-Fn2 describing historic/forecast days')

    din = int(din[1:])
    dout = int(dout[1:])

    if not din > 0 and dout > 0:
        raise ValueError('Num of historic input or of forecast days not greater than zero. Received {}, {}'.format(
            din, dout))

    model_dir = os.path.join('medium_term_ice_forecasting/datasets', directory)
    dataset_directory = resource_filename('sifnet', model_dir)
    print(dataset_directory)

    dataset_descriptor = split_path[-3]
    dataset_descriptor = dataset_descriptor + '.yaml'

    if dataset_descriptor not in os.listdir(dataset_directory):
        raise ValueError('Specified dataset {} does not exist in dataset directory'.format(dataset_descriptor))

    if type(visualize_forecasts) != str:
        raise TypeError('Received ')

    if visualize_forecasts not in {'all', 'none', 'best'}:
        raise ValueError('visualize_forecasts not one of valid options. Options are [\'all\', \'none\', \'best\'], '
                         'instead received {}'.format(visualize_forecasts))
    # done validating inputs

    model_names = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]

    model_paths = []
    temp = []
    for i in range(len(model_names)):
        name = model_names[i]
        model_path = os.path.join(path, name, "{}.h5".format(name))
        if os.path.exists(model_path):
            model_paths.append(model_path)
            temp.append(name)

    model_names = temp

    if val_years is None:
        val_years = [1989, 2000, 2002, 2014, 2016]

    if test_years is None:
        test_years = [1990, 2003, 2012, 2013, 2017]

    with open(os.path.join(dataset_directory, dataset_descriptor), 'r') as f:
        config = yaml.safe_load(f)
        dsm = DatasetManager(config)

    dsm.config(din, dout, validation_years=val_years, test_years=test_years)

    # TODO: parameterize

    outputs = [future_single_channel_thresholded(dsm)]

    if visualize_forecasts == 'all':
        viz = 1
    else:
        viz = 0

    restrict_to_available_gpu(1)
    best_val_acc = 0
    best = -1
    try:
        for i in range(len(model_names)):
            print("Starting model: ", model_names[i])
            inputs = [historical_all_channels(dsm)]
            if 'augmented' in model_names[i] or '_fc_' in model_names[i]:
                inputs.append(future_multi_channel(dsm, [2, 3, 4]))

            model_path = model_paths[i]
            model_name = model_names[i]
            model = load_model(model_path, custom_objects=custom_objects)
            model_dir_path = os.path.join(path, model_name)
            completed_without_crash = False
            attempts = 0
            while not completed_without_crash and attempts < 5:
                try:
                    metrics = evaluation_procedure(dsm, model, model_name, inputs, outputs, model_dir_path, path,
                                                   visualize_forecasts=viz, restrict_to_gpu=False)
                except Exception as e:
                    attempts = attempts + 1
                    if attempts < 5:
                        print(e)
                        print('Trying again...')
                    else:
                        cuda.close()
                        raise e
                else:
                    completed_without_crash = True

            if 'summary.yaml' in os.listdir(model_dir_path):
                with open(os.path.join(model_dir_path, 'summary.yaml'), 'r+') as f:
                    summary = yaml.safe_load(f)
                    if summary is None:
                        summary = dict()
                    summary['evaluation_metrics'] = metrics
                    yaml.safe_dump(summary, f)
            else:  # create summary if it doesn't exist
                with open(os.path.join(model_dir_path, 'summary.yaml'), 'x') as f:
                    summary = dict()
                    summary['evaluation_metrics'] = metrics
                    yaml.safe_dump(summary, f)

            val_acc = metrics['val_metrics']['accuracy']['mean']
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best = i

        if visualize_forecasts == 'best' and best != -1:
            model_path = model_paths[i]
            model_name = model_names[i]
            model = load_model(model_path, custom_objects=custom_objects)
            model_dir_path = os.path.join(path, model_name)
            print('Best model in directory is {} with {} val accuracy'.format(model_name, best_val_acc))
            print('Visualizing best model\'s forecasts')
            evaluation_procedure(dsm, model, model_name, inputs, outputs, model_dir_path, path,
                                 visualize_forecasts=1, restrict_to_gpu=False, compute_metrics=False,
                                 save_predictions=save_best_model_predicitons, close_gpu=True)

    finally:
        cuda.close()


def resume_training(name, save_path, config_name, max_epochs, batch_size,
                    val_years=[1989, 2000, 2002, 2014, 2016], test_years=[1990, 2003, 2012, 2013, 2017],
                    n_input_days=3, n_forecast_days=30, augmented=True, dataset_path='default'):
    """
    Function to continue training given model checkpoint. Model checkpoint should be named 'checkpoint.h5'.
    Saves the completed model to the same directory as 'checkpoint.h5'

    :param name: string
        name of .h5 file (filename of finished model)
    :param save_path: string
        where the 'checkpoint.h5' is saved
    :param config_name: string
        name of the yaml config file, along with its parent directory
        i.e. 'Hudson/Hudson_Breakup_v1.yaml'
    :param max_epochs: int
        maximum number of epochs we want to train
    :param val_years: list
        specifies the set of years used for validation
    :param test_years: list
        specifies the set of years used for testing
    :param batch_size: int
        batch size
    :param n_input_days: int
        number of historical days observed
    :param n_forecast_days: int
        number of days to be forecasted
    :param augmented:
        if True, model is augmented by forecasts of v10, u10 and t2m
    :param dataset_path:
        where config_name is stored
        default is under sifnet/medium_term_ice_forecasting/datasets
    """

    if dataset_path == 'default':
        dataset_path = resource_filename("sifnet", "medium_term_ice_forecasting/datasets")

    path_to_dataset_config = os.path.join(dataset_path, config_name)

    with open(path_to_dataset_config, 'r') as file:
        config = yaml.safe_load(file)
        file.close()

    dataset_manager = DatasetManager((config, path_to_dataset_config))
    dataset_manager.config(n_input_days, n_forecast_days, validation_years=val_years, test_years=test_years)

    model_inputs = [historical_all_channels(dataset_manager)]

    if augmented:
        model_inputs.append(future_multi_channel(dataset_manager, [2, 3, 4]))

    model_outputs = [future_single_channel_thresholded(dataset_manager)]

    (train_gen, train_steps_per_epoch) = dataset_manager.make_generator(batch_size, 'train',
                                                                        input_functions=model_inputs,
                                                                        output_functions=model_outputs,
                                                                        multithreading=False,
                                                                        safe=False)

    (val_gen, val_steps_per_epoch) = dataset_manager.make_generator(batch_size, 'val',
                                                                    input_functions=model_inputs,
                                                                    output_functions=model_outputs,
                                                                    multithreading=False,
                                                                    safe=False)
    checkpoint_name = 'checkpoint.h5'
    checkpoint_path = os.path.join(save_path, checkpoint_name)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1,
                                                    save_best_only=True)
    callbacks = [checkpoint]
    model = load_model(checkpoint_path)
    model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch,
                        epochs=max_epochs, validation_data=val_gen, validation_steps=val_steps_per_epoch,
                        callbacks=callbacks, max_queue_size=0)

    model = tf.keras.models.load_model(checkpoint_path)
    os.remove(checkpoint_path)

    model_path = os.path.join(save_path, "{}.h5".format(name))
    model.save(model_path)


def create_unified_years_array(savepath, years, filename, data_type, out_dir):

    """
    Merges the test array generated by each year without using dataset manager

    :param savepath: str
            path to a directory containing models evaluated at each year
            e.g /work/IcePresence/NWT_Freeze_v2/H3-F30/model_name
    :param years: list of str
            list of years, each year has a subdir in savepath
    :param filename: string
            name of npy file to be merged
            e.g. test_climate_normals.npy
    :param data_type: numpy dtype
            datatype the numpy array should be saved as
    :param out_dir
    :return: ndarray
            array containing aggregated data cube
    """
    
    test_records_path = os.path.join(savepath, str(years[0]), "test_record.yaml")

    with open(test_records_path, 'r') as file:
        test_records = yaml.safe_load(file)
        file.close()

    days_of_historic_input = test_records["days_of_historic_input"]
    forecast_days_forward = test_records["forecast_days_forward"]
    begin = test_records["yearly_start"]
    end = test_records["yearly_end"]

    year_to_cube = dict()
    dim = np.array([0, 0, 0, 0])
    
    for year in years:
        # years.extend(test_records[i])
        test_pred_path = os.path.join(savepath, str(year), filename)
        test_date_path = os.path.join(savepath, str(year), "aggregated_test_pred_dates.csv")

        predictions = np.load(test_pred_path)

        dim[0] += predictions.shape[0]
        dim[1:4] = predictions.shape[1:4]

        # whether or not yearly end goes into the next year
        wrap_around = bool(end[0] < begin[0] or (end[0] == begin[0] and end[1] < begin[1]))

        with open(test_date_path, 'r') as csv_file:
            # skip first line
            next(csv_file)
            csv_reader = csv.reader(csv_file, delimiter=',')

            index_of_first_day = 0
            starting_year_of_period = -1
            for row in csv_reader:
                current_year = int(row[0][:4])
                current_month = int(row[0][5:7])
                current_day = int(row[0][8:])

                current_date = date(current_year, current_month, current_day)
                # find beginning and end of year containing current_date
                if wrap_around and current_month >= begin[0]:
                    end_date = date(current_year + 1, end[0], end[1])
                else:
                    end_date = date(current_year, end[0], end[1])
                if wrap_around and current_month < begin[0]:
                    begin_date = date(current_year - 1, begin[0], begin[1])
                else:
                    begin_date = date(current_year, begin[0], begin[1])

                delta_forcasts = timedelta(forecast_days_forward + 1)
                delta_historic = timedelta(days_of_historic_input - 1)

                # found and record first day of year
                if current_date == begin_date + delta_historic:
                    index_of_first_day = int(row[1])
                    starting_year_of_period = int(row[0][:4])
                # found last day of year, slice data cube and save into dictionary
                elif current_date == end_date - delta_forcasts:
                    index_of_last_day = int(row[1])
                    year_to_cube[starting_year_of_period] = predictions[index_of_first_day:index_of_last_day + 1, :, :,
                                                            :]
    
    aggregated_filename = filename
    aggregated_file_path = os.path.join(out_dir, aggregated_filename)
    
    # if exists, create memory map of existing file
    if os.path.exists(aggregated_file_path):
        aggregated_cube = np.memmap(aggregated_file_path, shape=(dim[0], dim[1], dim[2], dim[3]),
                                    mode='r+', dtype=data_type)
    
    # if doesn't exist, initialize memory mapped array
    else:
        aggregated_cube = np.memmap(aggregated_file_path, shape=(dim[0], dim[1], dim[2], dim[3]),
                                        mode='w+', dtype=data_type)
        prev = 0
        cur = 0
        # aggregate data cubes at each year together in chronological order
        for year in years:
            ytc = year_to_cube[year]
            cur += ytc.shape[0]
            aggregated_cube[prev:cur] = ytc
            prev = cur

        # flush memory changes to disk
        aggregated_cube.flush()

        # load the aggregated numpy file as a read and write memmap
        aggregated_cube = np.memmap(aggregated_file_path, shape=(dim[0], dim[1], dim[2], dim[3]),
                                        mode='r+', dtype=data_type)
    
    return aggregated_cube


def aggregate_dates_years(savepath,years,out_dir):
    """
    Aggregates the dates saved from each year together 

    :param savepath: str
            path to a directory containing models evaluated at each fold
            e.g /work/IcePresence/NWT_Freeze_v2/H3-F30/model_name
    :param years: list
    
    """
    all_dates = []
    # iterate over 
    for year in years:
        test_pred_dates = os.path.join(savepath, str(year), "aggregated_test_pred_dates.csv")
        with open(test_pred_dates, 'r') as f:
            reader = csv.reader(f)
            dates_and_index = map(tuple, reader)

            dates = [x[0] for x in dates_and_index]
            dates.pop(0)
            all_dates.extend([date(int(x[:4]), int(x[5:7]), int(x[8:])) for x in dates])

    all_dates.sort()
    date_strings = [str(d) for d in all_dates]
    aggregated_dates_path = os.path.join(out_dir, "aggregated_test_pred_dates.csv")
    with open(aggregated_dates_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'numpy_join_index'])
        for i in range(len(date_strings)):
            d = date_strings[i]
            writer.writerow([d, i])


def months_days_heatmap(gen_path, landmask, region, model_keys, metric='acc', forecast_inputs=90, 
                        years_interval=[1996,2019], load=True):
    """
    Create the heatmap visualization as different months vs forecast length
    gen_path: str, 
    landmask: ndarray. Array containing the landmask
    region: str, Region of interest
    model_keys: list of str, list of model names to plot the heatmap for
    forecast_inputs: int, length of forecast
    years_interval: list of 2 ints, test years period
    load: wether to load available data or regenerate new ones
    """
    #check the metric and set the figures label, colormap, and min & max of colorbars
    if metric == 'acc':
        label = 'Accuracy'
        vmin = 0.8
        vmax = 1.0
        d_vmin = -0.02
        d_vmax = 0.1
        cmap = None
    elif metric == 'brier':
        label = 'Score'
        vmin = 0.0
        vmax = 0.2
        d_vmin = -0.02
        d_vmax = 0.01
        cmap="rocket_r"
    else:
        raise ValueError(f"Metric type '{metric}' is not supported")
    
    # create the output directory if it doesn't exist
    outpath = os.path.join(gen_path, region, 'heat_maps')
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # climate normal doesn't differ between models, so use just one
    filenames_with_model_keys = [("test_climate_normals", model_keys[0])]

    # want to get prediction data for each model type
    for model_key in model_keys:
        filenames_with_model_keys.append(("test_preds", model_key))
    
    data = dict()
    for zipped in filenames_with_model_keys:

        filename = zipped[0]
        model_key = zipped[1]

        target_name = f"monthly_{filename}_{metric}.npy"

        # create the path 
        if filename == "test_climate_normals":
            data_path = os.path.join(outpath,target_name)
            dtype = np.uint8

        elif filename == "test_preds":
            data_path = os.path.join(outpath, model_key+'_'+target_name)
            dtype = np.float16

        # load existing data file or create a new one
        if not load or not os.path.exists(data_path):    
            all_pred = np.zeros((12,forecast_inputs))

            # iterate over every month
            for mon in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']:
                mon_path = os.path.join(gen_path, region, f"{region}_{mon}", f"H3-F{forecast_inputs}")
                
                # access the aggregation directory for a specific model architecture
                savepath = max(glob.glob(os.path.join(mon_path, model_key+'*')), key=os.path.getmtime)
                savepath = os.path.join(savepath, str(years_interval[0])+'-'+str(years_interval[1]))

                # load climate normal/prediction data from aggregation directory
                try:
                    test_pred = np.load(os.path.join(savepath,"aggregated_{}.npy".format(filename)))
                    test_data = np.load(os.path.join(savepath,"aggregated_test-data.npy"))
                except:
                    forecast_dates = get_dates_from_csv(savepath)
                    shape=(len(forecast_dates),forecast_inputs,landmask.shape[0],landmask.shape[1])
                    test_pred = np.memmap(os.path.join(savepath,"aggregated_{}.npy".format(filename)),dtype=dtype,
                                          shape=shape)
                    test_data = np.memmap(os.path.join(savepath,"aggregated_test-data.npy"),dtype=np.uint8,
                                          shape=shape)

                forecast_dates = get_dates_from_csv(savepath)
                valid_unique_months, monthly_lookup_tables = unique_forecast_months_with_lookup_tables(forecast_dates,
                                                                                                       forecast_inputs)
                ind = list(calendar.month_abbr).index(mon)
                ind_f = (ind+3)
                if ind_f>12:
                    ind_f %= 12 #find the prediction month by adding 3 to the train month 
                table = monthly_lookup_tables[ind_f]

                relevant_truth = test_data[table] # extract the relevant samples
                relevant_pred = test_pred[table]
                
                if metric == 'acc':
                    (_,_),(_,_),(_,pred_score) = standard_evaluate(relevant_truth, relevant_pred, landmask,verbose=False)
                elif metric == 'brier':
                    (_,pred_score) = np_brier(relevant_truth, relevant_pred, landmask)

                all_pred[ind-1,:] = pred_score

                np.save(data_path,all_pred)

        # load
        else:
            all_pred = np.load(data_path)

        if filename == "test_climate_normals":
            data['climate_normal'] = all_pred
        else:
            data[model_key] = all_pred
    
    # data: [normals, basic, augmented]

    # mpl.rcParams['font.serif'] = ['Times New Roman']
    # sns.set_style({'font.family': 'Times New Roman'})
    # !rm ~/.cache/matplotlib -rf
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Helvetica']
    plt.rcParams['font.size'] = 12
    ax1 = sns.heatmap(data['climate_normal'], cbar_kws={'label': f'{label}'},vmin=vmin,vmax=vmax,cmap=cmap)
    ax1.set_yticklabels(list(calendar.month_abbr)[1:],rotation=0)
    ax1.set_xlabel('Forecast Lead Day',labelpad=10)
    ax1.set_ylabel('Month')
    plt.suptitle('Heat map of climate normals')
    plt.tight_layout()

    plt.savefig(os.path.join(outpath,f"climate_{metric}.eps"))
    plt.savefig(os.path.join(outpath,f"climate_{metric}.png"))
    plt.figure()
    for i in range(len(model_keys)):
        model_key = model_keys[i]

        # Plot the heatmap for the predictions themselves
        ax2 = sns.heatmap(data[model_keys[i]], cbar_kws={'label': f'{label}'},vmin=vmin,vmax=vmax,cmap=cmap)
        ax2.set_yticklabels(list(calendar.month_abbr)[1:],rotation=0)
        ax2.set_xlabel('Forecast Lead Day',labelpad=10)
        ax2.set_ylabel('Month')
        plt.suptitle(f'{label} Heat map of predictions')
        plt.tight_layout()
        plt.savefig(os.path.join(outpath,f"pred_{model_key}_{metric}.eps"))
        plt.savefig(os.path.join(outpath,f"pred_{model_key}_{metric}.png"))
        plt.figure()

        # Plot the heatmap for the difference between model and climate normal
        # First the basic model and then the augmented model
        ax3 = sns.heatmap(data[model_keys[i]]-data['climate_normal'], cbar_kws={'label': f'$\Delta$ {label}'},
                          vmin=d_vmin,vmax=d_vmax)
        ax3.set_yticklabels(list(calendar.month_abbr)[1:],rotation=0)
        ax3.set_xlabel('Forecast Lead Day',labelpad=10)
        ax3.set_ylabel('Month')
        plt.suptitle(f'{label} Heat map of predictions-normal difference')
        plt.tight_layout()
        plt.savefig(os.path.join(outpath,f"diff_{model_key}_climate_{metric}.eps"))
        plt.savefig(os.path.join(outpath,f"diff_{model_key}_climate_{metric}.png"))
        plt.figure()

    # Plot the heatmap for the difference between 
    if len(model_keys)>1:
        ax4 = sns.heatmap(data[model_keys[1]]-data[model_keys[0]], cbar_kws={'label': f'$\Delta$ {label}'},
                          vmin=d_vmin,vmax=d_vmax)
        ax4.set_yticklabels(list(calendar.month_abbr)[1:],rotation=0)
        ax4.set_xlabel('Forecast Lead Day',labelpad=10)
        ax4.set_ylabel('Month')
        plt.suptitle(f'{label} Heat map of augmented-basic model difference')
        plt.tight_layout()
        plt.savefig(os.path.join(outpath,f"diff_{model_keys[1]}_{model_keys[0]}_{metric}.eps"))
        plt.savefig(os.path.join(outpath,f"diff_{model_keys[1]}_{model_keys[0]}_{metric}.png"))


def plot_timeseries_for_points(savepath, locs, loc_coords, chosen_years, lead_day, forecast_days_forward, days_of_historic_input, model, threshold=True):
    '''
    Plot time series for chosen locations in the raster

    :param locs: list
        list of row-column indexes for each location 
    :param loc_coords: list
        list of latitude-longitude coordinate tuples associated with each location 
    :param chosen_years: list
        list of years (as integers) to consider
    :param lead_day: int
    :param threshold: bool
        True/False value for whether to threshold the SIC vlaues
    '''

    aggregated_mod_filename = "aggregated_test_preds.npy"
    aggregated_obs_filename = "aggregated_test-data.npy"
    aggregated_norm_filename = "aggregated_test_climate_normals.npy"
    pred_dates_csv_filename = "aggregated_test_pred_dates.csv"

    _, _, begin, end, _ = extract_test_records(savepath)

    if model == "modeled":
        preds_data = np.load(os.path.join(savepath, aggregated_mod_filename))
        title = "model_ts_for_"
        fig_title = "Model time series"
    elif model == "normals":
        preds_data = np.load(os.path.join(savepath, aggregated_norm_filename))
        title = "climate_normal_ts_for_"
        fig_title = "Normal time series"
    else:
        raise ValueError(f"Model type '{model}' is not supported")
    obs_data = np.load(os.path.join(savepath, aggregated_obs_filename))

    years_to_indices = find_indices_in_year(savepath, begin, end, forecast_days_forward=forecast_days_forward, 
                                            days_of_historic_input=days_of_historic_input)
    preds_at_lead_day = extract_gridded_daily_time_series(preds_data, lead_day)
    obs_at_lead_day = extract_gridded_daily_time_series(obs_data, lead_day)

    pred_dates_csv_path = os.path.join(savepath, pred_dates_csv_filename)
    index_to_dates = {}
    with open(pred_dates_csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            year, month, day = row[0].split("-")
            index_to_dates[int(row[1])] = date(int(year), int(month), int(day))

    chosen_years.sort()
    for j, loc_index in enumerate(locs):
        plt.rcParams['figure.figsize'] = (20, 8)
        fig, axs = plt.subplots(len(chosen_years), 2)
        for k, choice_year in enumerate(chosen_years):
            relevant_indices = years_to_indices[choice_year]
            relevant_indices.sort()
            # Shift start dates by historic input days and lead day to get actual prediction dates from launch dates
            dates_for_plot = [index_to_dates[d] + timedelta(days_of_historic_input) + timedelta(lead_day) 
                              for d in relevant_indices]
            mod_ts = preds_at_lead_day[relevant_indices, loc_index[0], loc_index[1]]
            obs_ts = obs_at_lead_day[relevant_indices, loc_index[0], loc_index[1]]
            #Thresholding
            thresholding_value = 0.5
            if threshold:
                mod_ts[mod_ts >= thresholding_value] = 1
                mod_ts[mod_ts < thresholding_value] = 0

            if axs.ndim == 1:
                mod_axes = axs[0]
                obs_axes = axs[1]
            # axs is two-dimensional
            else:
                mod_axes = axs[k][0]
                obs_axes = axs[k][1]
            # Plot prediction time series
            mod_axes.set(xlabel=f"Date range for year {choice_year}")
            mod_axes.set(ylabel=f"Pred. value")
            mod_axes.plot(dates_for_plot, mod_ts)
            # Plot observed time series
            obs_axes.set(xlabel=f"Date range for year {choice_year}")
            obs_axes.set(ylabel=f"Obs. value")
            obs_axes.plot(dates_for_plot, obs_ts)

        lon = round(loc_coords[j][0], 2)
        lat = round(loc_coords[j][1], 2)
        fig.suptitle(f"{fig_title} at location ({lon},{lat})", fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(savepath, 'evaluations', 'N_day_accuracy_map', title+f"({lon},{lat})" + '.png'))
        plt.close(fig)


if __name__ == "__main__":

    savepath = '/home/nazanin/workspace/local_data/results/IcePresence/Hudson/Hudson_Monthly/H3-F90/' \
               'spatial_feature_pyramid_net_hiddenstate_ND_cross_val_res_2020-12-15-18:03:52'

    years_interval = [2006,2017]
    years = list(range(years_interval[0],years_interval[1]+1))

    outpath = os.path.join(savepath, str(years_interval[0])+'-'+str(years_interval[1]))
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    ytrue = create_unified_years_array(savepath, years, "aggregated_test-data.npy", np.uint8, outpath)
    ypredict = create_unified_years_array(savepath, years, "aggregated_test_preds.npy", np.float16, outpath)
    normal = create_unified_years_array(savepath, years, "aggregated_test_climate_normals.npy", np.uint8, outpath)
    persistence = create_unified_years_array(savepath, years, "aggregated_test_persistence.npy", np.uint8, outpath)

    aggregate_dates_years(savepath, years, outpath)

    with open(resource_filename('sifnet', 'medium_term_ice_forecasting/datasets/Hudson/Hudson_Monthly.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        dsm = DatasetManager(config)
        dsm.forecast_days_forward = 90
    model_name = 'spatial_feature_pyramid_net_hiddenstate_ND'
    calculate_kfold_metrics(dsm, outpath, model_name)

    # create plots
    generate_graphs = True
    if generate_graphs:
        from sifnet.medium_term_ice_forecasting.utilities.standardized_evaluations import per_month_accuracy_plots, \
            per_month_accuracy_maps
        landmask = dsm.raw_data[0, :, :, dsm.find_landmask_channel()]
        dates = get_dates_from_csv(outpath)
        region = 'Hudson Bay'
        # plot monthly accuracy plots and maps
        lat_bounds = dsm.lat_bounds
        long_bounds = dsm.long_bounds

        outpath2 = os.path.join(outpath, 'evaluations', 'monthly_accuracy_maps')
        if not os.path.exists(outpath2):
            os.makedirs(outpath2)
        # plot monthly accuracy maps
        per_month_accuracy_maps(ytrue=ytrue, ypredict=ypredict, forecast_dates=dates, mask=landmask,
                                savepath=outpath2,
                                region_name=region, climate_norm=normal, lat_bounds=lat_bounds,
                                lon_bounds=long_bounds)

        outpath2 = os.path.join(outpath, 'evaluations', 'monthly_accuracy_plots')
        if not os.path.exists(outpath2):
            os.makedirs(outpath2)
        # plot monthly accuracy plots
        per_month_accuracy_plots(ytrue=ytrue, ypredict=ypredict, forecast_dates=dates, mask=landmask,
                                 savepath=outpath2,
                                 region_name=region, climate_norm=normal, persistence=persistence)

    print("hi")



