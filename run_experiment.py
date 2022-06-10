"""
This experiment file has been used to quickly run experiments on various regions with the latest
models and hyperparameters.

"""

import gc
from numba import cuda
import os, glob
import numpy as np
from copy import copy

from tensorflow.python.keras.backend import clear_session

import sifnet.utilities.gpu as gpu
from sifnet.medium_term_ice_forecasting.ice_presence.ice_presence_experiment import IcePresenceExperiment
from sifnet.data.GeneratorFunctions import historical_all_channels, future_multi_channel_climate, historical_multi_channel_climate


def run_experiment_month(region, base_path, mon, start_year, forecast_length, model_key, source, pre_computed_path):
    '''
    Run an experiment for a given month and start year
    :param region: Region of interest
    :param base_path: Results directory
    :param mon: Month of interest
    :param start_year: First desired test year
    :param forecast_length: Length of forecast
    :param model_key: Model key
    :param source: Source directory
    :param pre_computed_path: Path to computed input variables
    '''

    # number of gpus to use
    n_gpu = 1

    gpu.restrict_to_available_gpu(n_gpu)
    # gpu.set_gpu_options(kind="growth")

    try:
        # choose dataset yaml based on region/month
        dataset = region+'/'+region+'_'+mon+'.yaml'

        print('Chosen dataset: {}'.format(dataset))
        print('Forecast length: {}'.format(forecast_length))
        
        # init_year -- number of years to use for initial training
        initial_training_end = 1996

        # if start year 
        if start_year <= initial_training_end:
            init_year = 10
        else:
            init_year = start_year-initial_training_end+9

        # initialize experiment object for running pipeline
        exp = IcePresenceExperiment(os.path.join(base_path))

        # set up configurations and create/process datasets before running experiment
        dsm = exp.configure_experiment(dataset=dataset, n_input_days=3, n_forecast_days=forecast_length,
                                       source=source, pre_computed_path=pre_computed_path)

        # create a copy of experiment configuration for climate normal creation        
        dsm2 = copy(dsm)

        # define test, validation, training years for climate normal
        test_year = 1996
        validation_years = [test_year-1]
        test_years = [test_year]
        train_years = dsm2.years[:dsm.years.index(test_year-1)]

        dsm2.config(days_of_historic_input=3, forecast_days_forward=forecast_length, validation_years=validation_years,
                    test_years=test_years, train_remainder=False, custom_train_years=train_years)
        
        # define forecast channels
        forecast_channels = list(np.arange(8))#[1]#[2, 4, 5]
        
        # training hyperparameters
        training_kwargs = dict(num_gpu=n_gpu, max_epochs=60, bs_per_gpu=1, patience=5,
                            use_tensorboard=False, lr_decay=1e-4, initial_lr=0.01, optimizer='SGD',
                            monitor='val_loss',safe_gen=False)
        
        # model parameters  &
        # define what data will be available to model for prediction
        if model_key == 'spatial_feature_pyramid_net_hiddenstate_ND':
            model_kwargs = dict(l2reg=0.0003, 
                                input_shape=(dsm.days_of_historic_input, dsm.resolution[0], dsm.resolution[1],
                                             dsm.num_vars),
                                output_steps=forecast_length)
            exp.set_inputs([historical_all_channels(dsm)])
        elif model_key == 'spatial_feature_pyramid_anomaly':
            model_kwargs = dict(l2reg=0.0003, 
                                input_shape=(dsm.days_of_historic_input, dsm.resolution[0], dsm.resolution[1],
                                             dsm.num_vars),
                                anomaly_shape=(dsm.days_of_historic_input, dsm.resolution[0], dsm.resolution[1],
                                             len(forecast_channels)),
                                output_steps=forecast_length)
            exp.set_inputs([historical_all_channels(dsm),
                            historical_multi_channel_climate(dsm, channels=forecast_channels, option='train', num_days=3)])
        elif model_key == 'spatial_feature_pyramid_hidden_ND_fc':
            model_kwargs = dict(l2reg=0.0003,
                                input_shape=(dsm.days_of_historic_input, dsm.resolution[0], dsm.resolution[1],
                                            dsm.num_vars),
                                forecast_input_shape=(forecast_length, dsm.resolution[0], dsm.resolution[1],
                                                      len(forecast_channels)),
                                output_steps=forecast_length)
            exp.set_inputs([historical_all_channels(dsm), 
                            future_multi_channel_climate(dsm2, forecast_channels, num_days = forecast_length)])
        elif model_key == 'spatial_feature_pyramid_anomaly_fc':
            model_kwargs = dict(l2reg=0.0003,
                                input_shape=(dsm.days_of_historic_input, dsm.resolution[0], dsm.resolution[1],
                                             dsm.num_vars),
                                anomaly_shape=(dsm.days_of_historic_input, dsm.resolution[0], dsm.resolution[1],
                                             len(forecast_channels)),
                                forecast_input_shape=(forecast_length, dsm.resolution[0], dsm.resolution[1], 
                                                      len(forecast_channels)),
                                output_steps=forecast_length)
            exp.set_inputs([historical_all_channels(dsm),
                            historical_multi_channel_climate(dsm, channels=forecast_channels, option='train', num_days=3),
                            future_multi_channel_climate(dsm2, forecast_channels, num_days = forecast_length)])

        # start initial training if start year falls within initial training years
        if start_year<=1996:
            exp.run_monthly(init_years=10, model_key=model_key, training_kwargs=training_kwargs, model_kwargs=model_kwargs)

        # match using model key
        model_directory = max(glob.glob(os.path.join(exp.current_path, model_key+'*/')), key=os.path.getmtime)

        print('Current path is: {}'.format(exp.current_path))
        print('Which has model directory: {}'.format(model_directory))

        # retraining hyperparameters
        retraining_kwargs = dict(num_gpu=n_gpu, max_epochs=40, bs_per_gpu=1, patience=5,
                                 use_tensorboard=False, lr_decay=1e-4, initial_lr=0.01, optimizer='SGD',
                                 monitor='val_loss')

        # additional keyword arguments for retraining
        kwargs = dict(skip_evaluation=False, cross_validation=True, fold_dir="fold_0",
                      model_dir=model_directory, visualize_best=0, visualization_level=0, compute_metrics=True,
                      save_predictions=True)

        # Retrain the model
        for year in range(init_year, 35):
            exp.retrain_model(model_key=model_key, init_years=year, n_runs=1, training_kwargs=retraining_kwargs,
                              model_kwargs=model_kwargs, **kwargs)

        gc.collect()
        clear_session()
    finally:
        cuda.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, 
                                     description="Run an experiment for a given month and training year")

    parser.add_argument('--region', metavar='region', help='domain of the forecasting model', choices=['Hudson','NWT','Arctic','PanArctic'], default='Hudson')
    parser.add_argument('--results_dir', help='path to store results at')
    parser.add_argument('--month', metavar='month', help='month of experiment', choices=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], default='Apr')
    parser.add_argument('--year', type=int, help='year to start training from', default='1980')
    parser.add_argument('--forecast_length', type=int, help='number of days in forecast', default=90)
    parser.add_argument('--model_enum', metavar='model key', type=int, help='which model to use:'
                        '1: spatial_feature_pyramid_net_hiddenstate_ND,'
                        '2: spatial_feature_pyramid_hidden_ND_fc,'
                        '3: spatial_feature_pyramid_anomaly,'
                        '4: spatial_feature_pyramid_anomaly_fc',
                        choices=[1,2,3,4], default=1)
    parser.add_argument('--raw_data_source', help='path to raw data source')
    parser.add_argument('--pre_computed_vars', help='path for storing computed data variables')

    args = parser.parse_args()

    model_keys_enum = {1: 'spatial_feature_pyramid_net_hiddenstate_ND', 2:'spatial_feature_pyramid_hidden_ND_fc', 
                       3: 'spatial_feature_pyramid_anomaly', 4: 'spatial_feature_pyramid_anomaly_fc'}

    run_experiment_month(region=args.region, base_path=args.results_dir, mon=args.month, start_year=args.year,
                         forecast_length=args.forecast_length, model_key=model_keys_enum[args.model_enum],
                         source=args.raw_data_source, pre_computed_path=args.pre_computed_vars)

