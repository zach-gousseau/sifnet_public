"""
Evaluate model

More specifically, the  model is evaluated independently across the
validation and training sets. Then, the ensemble is evaluated as a whole using the mean-forecast
approach.

During this process, figures of 'performance per forecast day' are produced on
both the individual and collective ensemble.

Finally, if enabled, the set of forecasts as produced by the mean ensemble
are visualized for both datasets.

Notes
-----
- Forecasts can be computed and saved to disk or loaded from disk.
  - Two numpy array files are written: test and validation.

Computational
-------------
- Substantial memory requirements are needed.
- Run time for 4GB GPU running 33 years of data for Hudson Bay region from

Example
-------

"""

# FIXME: Add example to module docstring
# TODO: How to reduce length of import statements
# TODO: Add CLI progress bar -- Tensorflow provides own...
# TODO: Parameterize resolution

import os
import traceback
import csv

import numpy as np
import tensorflow as tf
from numba import cuda
from tqdm import tqdm
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as bk
from sifnet.data.DatasetManager import DatasetManager
import sifnet.medium_term_ice_forecasting.utilities.standardized_evaluations as se
import sifnet.medium_term_ice_forecasting.utilities.visualization as vf
from sifnet.data.GeneratorFunctions import climate_normal, persistence


def save_norms_and_persist(dsm, persist_path, norm_path, mode):
    """
    Merges the test array generated by each fold in cross validation.
    :param dsm: DatasetManager
            DatasetManager object needed to find number of dates in year
    :param persist_path: str
           path where persistence will be saved
    :param norm_path: str
           path where climate normal will be saved
    :param mode: str
           same options set for climate_normal
           Base normal on training set, training+validation sets, or all data.
           valid options are {'train', 'train+val', 'all'}
    """
    if mode == 'val':
        option = 'train'
    elif mode == 'test':
        option = 'train+val'

    dsm_gen, steps_per_epoch = dsm.make_generator(1, mode,
                                                  auxiliary_functions=[persistence(dsm, threshold=0.15),
                                                                       climate_normal(dsm, threshold=0.15,
                                                                                      option=option)])

    norms = np.ndarray(shape=(steps_per_epoch, dsm.forecast_days_forward,
                              dsm.resolution[0], dsm.resolution[1]),
                       dtype=np.uint8)

    persist = np.ndarray(shape=(steps_per_epoch, dsm.forecast_days_forward,
                                dsm.resolution[0], dsm.resolution[1]),
                         dtype=np.uint8)

    for i in tqdm(range(steps_per_epoch)):
        # print(i)
        (_, _, aux_outputs) = next(dsm_gen)
        in_persist = aux_outputs[0]
        in_norm = aux_outputs[1]
        # Accumulate from generator
        norms[i] = np.squeeze(in_norm)
        persist[i] = np.squeeze(in_persist)

    np.save(persist_path, persist)
    np.save(norm_path, norms)

    return persist, norms


def evaluation_procedure(dataset_manager, dsm2, model, model_name, model_inputs, model_outputs, output_directory,
                         pre_computed_directory, visualize_forecasts=0, restrict_to_gpu=True, compute_metrics=True,
                         save_predictions=False, cross_validation=False, **kwargs):
    """
    :param dataset_manager: DatasetManager
            The datasetManager which will be used to produce generators

    :param model: tf.keras.models.Model
            The model which will be evaluated

    :param model_name: string
            The name of this model

    :param model_inputs: list of tuples
            List of (generator_function, array_shape, datatype)
            These tuples are the return type of functions from GeneratorFunctions
            e.g. [historical_all_channels(dataset)]

    :param model_outputs: list of tuples
            List of (generator_function, array_shape, datatype)
            These tuples are the return type of functions from GeneratorFunctions
            e.g. [future_single_channel_thresholded(dataset)]

    :param output_directory: string, path-like.
            Figures will be saved to this directory.

    :param pre_computed_directory: string, path-like.
            Directory where pre-computed

    :param visualize_forecasts: int
            Specifies the level of forecast visualization
            0 -> no visualization
            1 -> test set visualization
            2 -> test and validation set visualization

    :param restrict_to_gpu: bool
            If true, gpu context will be restricted to 1 GPU. Default True.
            Only necessary if gpu context has not been restricted at a higher level.

    :param save_predictions: bool
            If true, model predictions will be saved to numpy file

    :param **kwargs: special keyword aguments
                - close_gpu: bool               Controls if GPU context should be destroyed while creating GIFs

    :return: metrics
            dictionary with keys(val_metrics, test_metrics)
                both are also dicts with keys (recall, precision, accuracy)
                    each is also a dict with keys (mean, daily)
    """

    if type(dataset_manager) != DatasetManager:
        raise TypeError('Received dataset_manger of non-DatasetManager type')

    if not dataset_manager.configured:
        raise ValueError('Received dataset_manager which has not been configured')

#     if type(model) != tf.keras.models.Model:
#         raise TypeError('Received model of non-tf.keras.models.Model type')

    if type(model_name) != str:
        raise TypeError('Received model_name of non-string type')

    if type(model_inputs) != list:
        raise TypeError('Received model_inputs of non-list type')

    if None in model_inputs:
        raise ValueError('Special value None found in model_inputs')

    if type(model_outputs) != list:
        raise TypeError('Received model_inputs of non-list type')

    if None in model_outputs:
        raise ValueError('Special value None found in model_inputs')

    if type(output_directory) != str:
        raise TypeError('Received output_directory of non-string type')

    if not os.path.exists(output_directory):
        raise ValueError('output_directory does not exist')

    if type(pre_computed_directory) != str:
        raise TypeError('Received pre_computed_directory of non-string type')

    if not os.path.exists(pre_computed_directory):
        raise IOError('pre_computed_directory does not exist')
    print('using pre_computed_directory {}'.format(pre_computed_directory))

    if type(visualize_forecasts) != int:
        raise TypeError('Received visualize_forecasts of non-int type')

    if visualize_forecasts < 0 or visualize_forecasts > 2:
        raise ValueError('Received visualize_forecasts of invalid value. '
                         'Received {}, must be 0, 1, or 2'.format(visualize_forecasts))

    close_gpu = kwargs.get('close_gpu', False)

    batch_size = 1
    # model._make_predict_function()

    if restrict_to_gpu:
        gpu.restrict_to_available_gpu(1)
    try:
        landmask_channel = dataset_manager.find_landmask_channel()
        landmask = dataset_manager.raw_data[0, :, :, landmask_channel]
        val_samples = len(dataset_manager.val_dates)
        test_samples = len(dataset_manager.test_dates)

        print("Starting processing.....")
        # #################################
        # EVALUATE/LOAD BASE TRUTH AND FIRST-DAY-ICE-CONDITIONS(for visualization)

        # ### VAL DATA!
        if val_samples > 0:
            validation_data_path = os.path.join(pre_computed_directory, "validation-data.npy")
            validation_first_days_ice_conditions_path = os.path.join(pre_computed_directory,
                                                                     "validation-first-days-ice-conditions.npy")
            dimensions_match = True
            paths_exist = False #os.path.exists(validation_data_path) and os.path.exists(
                #validation_first_days_ice_conditions_path)
            if paths_exist:
                validation_data = np.load(validation_data_path)
                validation_first_days_ice_conditions = np.load(validation_first_days_ice_conditions_path)
                val_d_shape = list(validation_data.shape)
                val_fdic_shape = list(validation_first_days_ice_conditions.shape)
                dimensions_match = bool(val_d_shape[0] == val_fdic_shape[0] and val_d_shape[-2:] == val_fdic_shape[-2:]
                                        and val_d_shape[-2:] == list(landmask.shape)
                                        and val_d_shape[1] == dataset_manager.forecast_days_forward
                                        and val_d_shape[0] == val_samples)
                del val_d_shape, val_fdic_shape
                if not dimensions_match:
                    print('WARNING: Loaded val data does not match expected dimensions! Re-computing!')
                else:
                    print('Info: Loaded val data successfully')

            if not dimensions_match or not paths_exist:
                print("Extracting validation base truth")
                (val_gen, val_steps_per_epoch) = dataset_manager.make_generator(batch_size, 'val')
                validation_data = np.ndarray(shape=(val_samples, dataset_manager.forecast_days_forward,
                                                    dataset_manager.resolution[0], dataset_manager.resolution[1]),
                                             dtype=np.uint8)

                # 'first day' ice conditions for validation set forecast visualization
                validation_first_days_ice_conditions = np.ndarray(shape=(val_samples, dataset_manager.resolution[0],
                                                                         dataset_manager.resolution[1]),
                                                                  dtype=np.float32)
                for i in tqdm(range(val_steps_per_epoch)):
                    # Create test and validation data to evaluate forecast
                    v_data, v_label = next(val_gen)

                    # v_data is a (batch_size, 3, 160, 300, 8) array inside an array of size 1
                    validation_data[i] = np.squeeze(v_label[0])
                    validation_first_days_ice_conditions[i] = np.squeeze(v_data[0][:, 2, :, :, 0])

                np.save(validation_data_path, validation_data)
                np.save(validation_first_days_ice_conditions_path, validation_first_days_ice_conditions)

            # only need to load/generate persist/norm if metrics are relevant
            if compute_metrics:
                # ##############################################
                # EVALUATE/LOAD PERSISTENCE and NORM
                val_persist_path = os.path.join(pre_computed_directory, 'val_persistence.npy')
                val_norm_path = os.path.join(pre_computed_directory, 'val_climate_normals.npy')
                dimensions_match = True
                paths_exist = False #os.path.exists(val_persist_path) and os.path.exists(val_norm_path)
                if paths_exist:
                    val_norms = np.load(val_norm_path)
                    val_persist = np.load(val_persist_path)
                    val_n_shape = list(val_norms.shape)
                    val_p_shape = list(val_persist.shape)
                    dimensions_match = bool(val_n_shape[0] == val_p_shape[0] and val_n_shape[0] == val_samples
                                            and val_n_shape[-2:] == val_p_shape[-2:]
                                            and val_n_shape[-2:] == list(landmask.shape)
                                            and val_n_shape[1] == dataset_manager.forecast_days_forward
                                            and val_n_shape[1] == val_p_shape[1])
                    del val_n_shape, val_p_shape
                    if not dimensions_match:
                        print("WARNING: Loaded val norms/persist does not match expected shape! Re-computing!")
                    else:
                        print("Info: Loaded val norms/persist successfully")

                if not dimensions_match or not paths_exist:  # files don't exist or don't match expectations
                    print("Extracting validation norms and persists")
                    val_persist, val_norms = save_norms_and_persist(dsm2,
                                                                    val_persist_path, val_norm_path, 'val')

                # past the nested if.
                print('val persistence metrics')
                val_persist_metrics = se.standard_evaluate(validation_data, val_persist, landmask, verbose=False)
                del val_persist
                print('val normal metrics')
                val_normal_metrics = se.standard_evaluate(validation_data, val_norms, landmask, verbose=False)
                del val_norms

            # always need to produce val predictions, either for metrics or for visualization (if val_samples > 0)
            (val_gen, val_steps_per_epoch) = dataset_manager.make_generator(batch_size, 'val',
                                                                            input_functions=model_inputs,
                                                                            output_functions=model_outputs)

            val_pred = np.ndarray(shape=(val_steps_per_epoch, dataset_manager.forecast_days_forward,
                                         dataset_manager.resolution[0], dataset_manager.resolution[1]),
                                  dtype=np.float32)
            # Generate this model's predictions
            print('Computing val predictions')
            for i in tqdm(range(val_steps_per_epoch)):
                (data, _) = next(val_gen)
                # val_pred[i] = model.predict_on_batch(data)
                # meant to solve tensor is not an element of this graph errors. Disable if necessary
                # moved to top of script
                # model._make_predict_function()
                val_pred[i] = np.squeeze(model.predict(data, batch_size=batch_size, steps=1, max_queue_size=0))

            del val_gen, data

            if save_predictions:
                np.save(os.path.join(output_directory, 'val_preds.npy'), val_pred)
                # write associated dates also
                with open(os.path.join(output_directory, 'val_pred_dates.csv'), 'w') as file:
                    writer = csv.writer(file)
                    date_strings = [str(d) for d in dataset_manager.val_dates]
                    writer.writerow(['date', 'numpy_join_index'])
                    for i in range(len(date_strings)):
                        d = date_strings[i]
                        writer.writerow([d, i])
                    del date_strings

            if compute_metrics:
                print('Computing val metrics:')
                val_metrics = se.standard_evaluate(validation_data, val_pred, landmask, verbose=False)
                all_val_metrics = [val_metrics, val_persist_metrics, val_normal_metrics]

        # ### TEST DATA!
        if test_samples > 0:
            if not cross_validation:
                test_data_path = os.path.join(pre_computed_directory, "test-data.npy")
                test_first_days_ice_conditions_path = os.path.join(pre_computed_directory,
                                                                   "test-first-days-ice-conditions.npy")
            else:
                test_data_path = os.path.join(output_directory, "test-data.npy")
                test_first_days_ice_conditions_path = os.path.join(output_directory,
                                                                   "test-first-days-ice-conditions.npy")

            dimensions_match = True
            paths_exist = os.path.exists(test_data_path) and os.path.exists(test_first_days_ice_conditions_path)
            if paths_exist:
                test_data = np.load(test_data_path)
                test_first_days_ice_conditions = np.load(test_first_days_ice_conditions_path)
                test_d_shape = list(test_data.shape)
                test_fdic_shape = list(test_first_days_ice_conditions.shape)
                dimensions_match = bool(test_d_shape[0] == test_fdic_shape[0] and test_d_shape[0] == test_samples
                                        and test_d_shape[1] == dataset_manager.forecast_days_forward
                                        and test_d_shape[-2:] == test_fdic_shape[-2:]
                                        and test_d_shape[-2:] == list(landmask.shape))
                del test_d_shape, test_fdic_shape
                if not dimensions_match:
                    print("WARNING: Loaded Test data does not match expected dimensions! Re-computing!")
                else:
                    print("Info: Loaded Test data successfully")

            if not dimensions_match or not paths_exist:  # files don't exist or don't match expectations
                print("Extracting test base truth")

                (test_gen, test_steps_per_epoch) = dataset_manager.make_generator(batch_size, 'test')

                test_data = np.ndarray(shape=(test_samples, dataset_manager.forecast_days_forward,
                                              dataset_manager.resolution[0], dataset_manager.resolution[1]),
                                       dtype=np.uint8)

                # 'first day' ice conditions for test set forecast visualization
                test_first_days_ice_conditions = np.ndarray(shape=(test_samples, dataset_manager.resolution[0],
                                                                   dataset_manager.resolution[1]),
                                                            dtype=np.float32)

                for i in tqdm(range(test_steps_per_epoch)):
                    t_data, t_label = next(test_gen)

                    test_data[i] = np.squeeze(t_label)
                    # grabbing [x, y] raster of init day sea ice concentration and placing in
                    # test_first_days_ice_conditions[i]
                    # thus test_first_days_ice_conditions has shape [len(test_steps_per_epoch), x, y]
                    test_first_days_ice_conditions[i] = np.squeeze(t_data[0][:, -1, :, :, 0])

                # Write test data arrays to disk for reuse
                np.save(test_data_path, test_data)
                np.save(test_first_days_ice_conditions_path, test_first_days_ice_conditions)

            if compute_metrics:
                # ##############################################
                # EVALUATE/LOAD PERSISTENCE and NORM
                if not cross_validation:
                    test_persist_path = os.path.join(pre_computed_directory, 'test_persistence.npy')
                    test_norm_path = os.path.join(pre_computed_directory, 'test_climate_normals.npy')
                else:
                    test_persist_path = os.path.join(output_directory, 'test_persistence.npy')
                    test_norm_path = os.path.join(output_directory, 'test_climate_normals.npy')

                dimensions_match = True
                paths_exist = os.path.exists(test_persist_path) and os.path.exists(test_norm_path)
                if paths_exist:
                    test_norms = np.load(test_norm_path)
                    test_persist = np.load(test_persist_path)
                    test_n_shape = list(test_norms.shape)
                    test_p_shape = list(test_persist.shape)
                    dimensions_match = bool(test_n_shape[0] == test_p_shape[0] and test_n_shape[0] == test_samples
                                            and test_n_shape[1] == test_p_shape[1]
                                            and test_n_shape[1] == dataset_manager.forecast_days_forward
                                            and test_n_shape[-2:] == test_p_shape[-2:]
                                            and test_n_shape[-2:] == list(landmask.shape))
                    del test_n_shape, test_p_shape
                    if not dimensions_match:
                        print('WARNING: Loaded Test norms/persist do not match expected shape! Re-Computing!')
                    else:
                        print('Info: Loaded Test norms/persists successfully')

                if not dimensions_match or not paths_exist:  # files don't exist or don't match expectations
                    print("Calculating climate normals and persistence")
                    # compute norms and persists
                    test_persist, test_norms = save_norms_and_persist(dsm2,
                                                                      test_persist_path, test_norm_path, 'test')

                print('test persistence metrics')
                test_persist_metrics = se.standard_evaluate(test_data, test_persist, landmask, verbose=False)
                del test_persist
                print('test normal metrics')
                test_normal_metrics = se.standard_evaluate(test_data, test_norms, landmask, verbose=False)
                del test_norms

                # ################################################################
                # DONE WITH PRE-PROCESSING, EVALUATE THIS PARTICULAR MODEL!

            (test_gen, test_steps_per_epoch) = dataset_manager.make_generator(batch_size, 'test',
                                                                              input_functions=model_inputs,
                                                                              output_functions=model_outputs)

            test_pred = np.ndarray(shape=(test_steps_per_epoch, dataset_manager.forecast_days_forward,
                                          dataset_manager.resolution[0], dataset_manager.resolution[1]),
                                   dtype=np.float32)
            print('Computing Test predictions')
            for i in tqdm(range(test_steps_per_epoch)):
                (data, _) = next(test_gen)
                # test_pred[i] = model.predict_on_batch(data)
                # meant to solve tensor is not an element of this graph errors. Disable if necessary
                # model._make_predict_function() # at top of script
                test_pred[i] = np.squeeze(model.predict(data, batch_size=batch_size, steps=1, max_queue_size=0))

            del test_gen, data

            if save_predictions:
                np.save(os.path.join(output_directory, 'test_preds.npy'), test_pred)
                # also write the dates associated with the predictions to file
                with open(os.path.join(output_directory, 'test_pred_dates.csv'), 'w') as file:
                    writer = csv.writer(file)
                    date_strings = [str(d) for d in dataset_manager.test_dates]
                    writer.writerow(['date', 'numpy_join_index'])
                    for i in range(len(date_strings)):
                        d = date_strings[i]
                        writer.writerow([d, i])
                    del date_strings

            if compute_metrics:
                print('Computing test metrics:')
                test_metrics = se.standard_evaluate(test_data, test_pred, landmask, verbose=False)
                all_test_metrics = [test_metrics, test_persist_metrics, test_normal_metrics]

        # PAST DATA GENERATION - Val and Test both processed (if available)!
        if compute_metrics:
            # TODO: Add to log
            print("\n")
            print("Model {} Characteristics".format(model_name))

            # val metrics and test metrics are triplets of tuples
            # (recall, precision, accuracy)
            # where each are a tuple of (mean, 30 day time series)
            plot_name = "Model_{}_evaluation.png".format(model_name)
            plot_path = os.path.join(output_directory, plot_name)

            # Performances of this particular model
            plot_text = "Metrics as a function of Forecast Day - Individual Model {}".format(model_name)

            if val_samples > 0 and test_samples > 0:
                se.standard_performance_plots(all_val_metrics, all_test_metrics,
                                              ['Modeled', 'Persistence', 'Climate Normal'],
                                              plot_path, plot_text)
            elif val_samples > 0:
                print('Metrics: Val set only')
                se.single_dataset_standard_performance_plots(all_val_metrics,
                                                             ['Modeled', 'Persistence', 'Climate Normal'],
                                                             plot_path, plot_text)
            else:
                print('Metrics: Test set only')
                se.single_dataset_standard_performance_plots(all_test_metrics,
                                                             ['Modeled', 'Persistence', 'Climate Normal'],
                                                             plot_path, plot_text)

        if visualize_forecasts > 0:  # visualization level 1. Test set only.
            if close_gpu:
                print('Closing CUDA When Visualizing!')
                cuda.close()  # would be better to find an alternative to this
            test_dir = os.path.join(output_directory, "test_vis")

            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            test_dates = dataset_manager.test_dates
            if test_samples > 0:
                vf.visualize_forecasts(test_pred, test_dates,
                                       test_first_days_ice_conditions, test_dir, test_data, mask=landmask,
                                       lat_bounds=dataset_manager.lat_bounds, lon_bounds=dataset_manager.long_bounds)
            else:
                print("Could not visualize test set. No data is available.")
                visualize_forecasts = visualize_forecasts + 1  # try to visualize val set even if set not to otherwise

            if visualize_forecasts > 1:  # visualization level 2. Test + Val sets.
                val_dir = os.path.join(output_directory, "val_vis")

                if not os.path.exists(val_dir):
                    os.mkdir(val_dir)

                val_dates = dataset_manager.val_dates
                if val_samples > 0:
                    vf.visualize_forecasts(val_pred, val_dates,
                                           validation_first_days_ice_conditions, val_dir, validation_data,
                                           mask=landmask, lat_bounds=dataset_manager.lat_bounds,
                                           lon_bounds=dataset_manager.long_bounds)
                else:
                    print("Could not visualize val set. No data is available.")

    except Exception as e:
        print(traceback.format_exc())
        print(e)
        cuda.close()
        raise Exception('Encountered an internal exception. See above error message.')

    print("Evaluation complete")

    metrics = dict()
    if compute_metrics:
        # prepare metrics into dictionaries
        # val metrics and test metrics are triplets of tuples
        # (recall, precision, accuracy)
        # where each are a tuple of (mean, 30 day time series)
        if val_samples > 0:
            val_recall = val_metrics[0]
            mean_val_recall = float(val_recall[0])
            daily_val_recall = [float(x) for x in val_recall[1]]  # type-casting for yaml

            val_precision = val_metrics[1]
            mean_val_precision = float(val_precision[0])
            daily_val_precision = [float(x) for x in val_precision[1]]

            val_accuracy = val_metrics[2]
            mean_val_accuracy = float(val_accuracy[0])
            daily_val_accuracy = [float(x) for x in val_accuracy[1]]
            metrics['val_metrics'] = dict(recall=dict(mean=mean_val_recall, daily=daily_val_recall),
                                          precision=dict(mean=mean_val_precision, daily=daily_val_precision),
                                          accuracy=dict(mean=mean_val_accuracy, daily=daily_val_accuracy))

        if test_samples > 0:
            test_recall = test_metrics[0]
            mean_test_recall = float(test_recall[0])
            daily_test_recall = [float(x) for x in test_recall[1]]

            test_precision = test_metrics[1]
            mean_test_precision = float(test_precision[0])
            daily_test_precision = [float(x) for x in test_precision[1]]

            test_accuracy = test_metrics[2]
            mean_test_accuracy = float(test_accuracy[0])
            daily_test_accuracy = [float(x) for x in test_accuracy[1]]

            metrics['test_metrics'] = dict(recall=dict(mean=mean_test_recall, daily=daily_test_recall),
                                           precision=dict(mean=mean_test_precision, daily=daily_test_precision),
                                           accuracy=dict(mean=mean_test_accuracy, daily=daily_test_accuracy))

    return metrics


if __name__ == "__main__":
    """
    A demonstration of how to use evaluation_procedure outside of the experiment pipeline
    """
    import tensorflow as tf
    from sifnet.data.DatasetManager import DatasetManager
    from pkg_resources import resource_filename
    from pprint import pprint as prettyprint
    from sifnet.data.GeneratorFunctions import historical_all_channels, future_single_channel_thresholded,future_multi_channel
    import sifnet.utilities.gpu as gpu
    import yaml
    import tensorflow.keras.layers as kl
    import tensorflow.keras.backend as bk
    from tensorflow import keras

    # load a dataset
    with open(resource_filename('sifnet', 'medium_term_ice_forecasting/datasets/Hudson/Hudson_Freeze_v2.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        dsm = DatasetManager(config)

    # configure our DatasetManager
    dsm.config(3, 90, validation_years=[1989, 2000, 2002, 2014, 2016], test_years=[1990, 2003, 2012, 2013, 2017])

    # load a trained model
    model_name = 'demo_model'
    # model = tf.keras.models.load_model('home/nazanin/workspace/local_data/results/IcePresence/NWT/NWT_Freeze_v2/H3-F30/leaky_baseline_30D_2020-01-07-20:54:07/leaky_baseline_30D_2020-01-07-20:54:07.h5')
    model = tf.keras.models.load_model('/home/nazanin/workspace/local_data/results/IcePresence/Hudson/Hudson_Freeze_v2/'
                                       'H3-F90/spatial_feature_pyramid_hidden_ND_fc_cross_val_res_2020-11-18-20:56:28/'
                                       'fold_0/spatial_feature_pyramid_hidden_ND_fc_2020-11-18-20:56:29.h5',
                                       custom_objects={'tf': tf,'kl':kl,'bk':bk,'keras':keras})
    # prepare our inputs & outputs
    inputs = [historical_all_channels(dsm), future_multi_channel(dsm, [2, 4, 5])]
    outputs = [future_single_channel_thresholded(dsm)]

    # declare paths
    save_dir = '/home/nazanin/workspace/local_data/results'
    pre_computed_dir = '/home/nazanin/workspace/local_data/results/IcePresence/Hudson/Hudson_Freeze_v2/H3-F90'  # pre-computed climate normals and persistence data

    # Finally, run the evaluation procedure
    # visualize_forecasts=1 means visualize test set only
    metrics = evaluation_procedure(dsm, model, model_name, inputs, outputs, save_dir,
                                   pre_computed_dir, visualize_forecasts=1, restrict_to_gpu=False)

    prettyprint(metrics)
