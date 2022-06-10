"""
The Experiment class enables one to run the entire ML pipeline from a single, simple script.
Task which are completed by Experiment include:
    - loading a dataset (Using DatasetManager)
    - configuration
    - Input/output relationship
    - Instatiating a model
    - Training a model (Using training_procedure)
    - Evaluation
    - K fold cross evaluation

The Experiment base class is purely virtual and includes a number of non-implemented methods.
These non-implemented methods are:
    - make_model
        Takes two arguments: a key and a set of keyword aguments **kwargs
        The key should be the name of a model architecture. The kwargs are passed into the function to create the
        specified model.
        Is not implemented because the valid models depend on the type of experiment i.e presence vs concentration
    - set_targets
        Takes no arguments. Is called at the end of the model initalization.
        This is used to determine what generator function will be used to produce the model target (supervised labels)
        which it is attempting to output based on the given inputs. Thus, it is type dependent.
    - evaluate
        Takes numerous arguments. In need of refactoring (note to future dev) to use kwargs instead.
        Each type of experiment must implement it's own evalation routine, because the type of experiment
        will determine how performance is measured.

The purpose of this forced subclassing (inheritance) is because the base Experiment class includes many different
functionalities which would be common across any different ML experiment, but there are also many different
aspects which must be customized based on the specific use case.

The basic sequence required to use experiment:

1) Instantiate an Experiment (subclass)
2) Configure to select a dataset, N input days, and N output days.
3) Set the inputs to be used for this experiment
4) define your hyperparameters and model parameters
5) experiment.run or experiment.run_k_fold

"""

import os
import copy
from typing import Dict, Any
import glob

import yaml
import datetime
import gc
import traceback
import numpy as np
from numba import cuda
from pkg_resources import resource_filename
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import clear_session

from sifnet.data.DatasetManager import DatasetManager as dsm
from sifnet.utilities.training_procedure import training_procedure
from sifnet.medium_term_ice_forecasting.utilities.standardized_evaluations import metrics_to_yaml_format
from sifnet.medium_term_ice_forecasting.utilities.postprocessing_tools import \
    create_unified_kfold_array, calculate_kfold_metrics, aggregate_dates, get_dates_from_csv
from sifnet.medium_term_ice_forecasting.utilities.standardized_evaluations import per_month_accuracy_plots, \
    per_month_accuracy_maps
from sifnet.data.GeneratorFunctions import historical_all_channels,future_multi_channel_climate, \
    historical_multi_channel_climate


class Experiment:
    """
    Base class which must be subclassed through inheritance.

    The class' two primary use-cases are run() and run_k_fold()

    run() is useful for model design and hyperparameter tuning. It trains a number of models
    on the given dataset and model inputs. The model output target must be specified by a experiment
    subclass and essentially defines whatever is attempting to be forecast. The train/val/test split must
    be configured explicitly per each run or may be defined by the experiment subclass.

    run_k_fold() takes the given dataset and splits it into k distinct non-overlapping groups.
    One group is selected as the test set, and the remainder are selected as the training set.
    A model is trained on the training set and then evaluated on the test set, using the normal run() routine.
    This process is repeated until each of the groups has been used as the test set exactly once.
    The evaluation metrics are averaged over each internal run so as to produce a highly robust estimate of model
    generalization performance.

    """
    minimum_performance_metrics = ...  # type: Dict[Any, Any]

    def __init__(self,base_path='./results'):
        self.name = 'BaseExperiment'
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        self.base_path = base_path
        self.current_path = ""  # defined in configure_experiment()
        self.inputs = [None]    # inputs available to our model
        self.targets = [None]   # whatever our model is trying to predict
        self.datasets_path = resource_filename("sifnet", "medium_term_ice_forecasting/datasets")
        self.dataset_name = ""
        self.available_datasets = self._get_available_datasets()
        self.dataset_manager = None  # placeholder for dataset manager objectF
        self._summary = None
        self.val_years = None   # Set by subclass, OR by config()
        self.test_years = None  # Set by subclass, OR by config()
        self.configured = False  # Set by subclass, OR by config()
        self.minimum_performance_metrics = dict()  # Set by subclass OR ignore minimum performance thresholds

    def _get_available_datasets(self):
        """
        Finds all available datasets in the dataset directory
        :return: list of available datasets
        """
        if self.datasets_path is None:
            return None

        path = self.datasets_path
        if not os.path.exists(path):
            raise ValueError('Path {} does not exist'.format(path))

        if not os.path.isdir(path):
            raise ValueError('Path {} is not a directory'.format(path))

        folders = []
        folder_paths = []

        # traverse dataset directory
        for r, d, f in os.walk(path):
            for folder in d:
                folders.append(folder)
                folder_paths.append(os.path.join(r, folder))

        files = []

        for i in range(len(folder_paths)):
            files.extend([os.path.join(folders[i], file) for file in os.listdir(folder_paths[i]) if file.endswith(".yaml")])

        return files

    def configure_experiment(self, dataset, n_input_days, n_forecast_days, val_years=None, test_years=None , source=None, pre_computed_path=None):
        """
        Configure the experiment prior to calling run() or run_k_fold().
        Information from this call is used to configure a DatasetManager object which is returned.
        Optional arguments val_years and test_years allow for these to be overridden if already set or initially set
            if working from a subclass which has not already established a default split.
        :param dataset: string
                    Name of the desired dataset e.g 'Hudson_Freeze_v2.yaml'
        :param n_input_days: int
                    The number of days of historic input
        :param n_forecast_days: int
                    The number of days to be forecasted
        :param val_years: List, optional
                    Custom validation years
        :param test_years: List, optional
                    Custom test years
        :return: DatasetManager
        """

        # Validate inputs and check that all necessary fields have been specified by a valid subclass
        if type(dataset) != str:
            raise TypeError('Received dataset of non-string type. Should be name of a valid dataset config.')

        if self.available_datasets is None:
            raise NotImplementedError('datasets_path must be specified by an experiment subclass')

        if dataset not in self.available_datasets:
            raise ValueError('Could not find specified dataset {} in available datasets {}'
                             .format(dataset, self.available_datasets))

        if type(n_input_days) != int:
            raise TypeError('Received n_input_days of non-int type')

        if n_input_days < 1:
            raise ValueError('Received invalid value of n_input_days,  must be greater than or equal to 1')

        if type(n_forecast_days) != int:
            raise TypeError('Received n_forecast_days of non-int type')

        if n_forecast_days < 1:
            raise ValueError('Received invalid value of n_forecast_days,  must be greater than or equal to 1')

        if val_years is not None:
            self.val_years = val_years
        if self.val_years is None:
            raise NotImplementedError('val_years must be specified by an experiment subclass or explicitly configured.')
        if test_years is not None:
            self.test_years = test_years
        if self.test_years is None:
            raise NotImplementedError('test_years must be specified by an experiment subclass or explicitly configured')

        self.dataset_name = dataset
        # DONE VALIDATING INPUTS
        path_to_dataset_config = os.path.join(self.datasets_path, dataset)
        with open(path_to_dataset_config, 'r') as file:
            config = yaml.safe_load(file)
            file.close()

        dataset_name_without_yaml = dataset[0:-5]  # string slicing, last 5 character are '.yaml', which we remove.
        input_output_relation = "H{}-F{}".format(n_input_days, n_forecast_days)
        self.current_path = os.path.join(self.base_path, dataset_name_without_yaml, input_output_relation)

        if (source is not None) and (pre_computed_path is not None):
            print(source)
            print(pre_computed_path)
            self.dataset_manager = dsm((config, path_to_dataset_config),source,pre_computed_path)
        else:
            self.dataset_manager = dsm((config, path_to_dataset_config))
        self.dataset_manager.config(n_input_days, n_forecast_days, validation_years=self.val_years,
                                    test_years=self.test_years)

        self.configured = True
        self.set_targets()  # must be after self.configured = True
        return self.dataset_manager

    def set_inputs(self, inputs):
        """
        Sets the inputs which will be provided to model.
        This defines what data will be available to the model in order to make its predictions

        :param inputs: list of tuples
                Each element of inputs must be a tuple created by a generator_function
        """
        if not self.configured:
            raise ValueError('Experiment has not been configured!')
        if type(inputs) != list:
            raise  TypeError('Received inputs of non-list type')
        if len(inputs) == 0:
            raise ValueError('Received inputs as empty list')
        for input in inputs:
            if type(input) != tuple or len(input) != 3:
                raise ValueError('Received invalid input {}. Each element of inputs must be tuple of length 3'
                                 .format(input))
        self.inputs = inputs

    def train(self, model, savepath, name, **train_kwargs):
        """
        Wrapper function for training_procedure

        :param model: keras.Model
        :param savepath: string path to a directory
        :param name: string, name of model
        :param train_kwargs: keyword arguments to be passed to training_procedure
        :return: model, training history
        """
        return training_procedure(model, self.inputs, self.targets, self.dataset_manager,
                                  savepath, name, restrict_to_gpu=False, **train_kwargs)

    def evaluate(self, model, model_name, savepath, visualize, compute_metrics=True, save_predictions=False,
                 cross_validation=False):
        """
        Evaluates the given model.

        Method must be overridden by Experiment subclass.


        :param model: tf.keras.models.Model object
        :param model_name: string
        :param savepath: string (path-like)
        :param visualize: int
                    0 -> no visualization
                    1 -> test set visualization
                    2 -> test and validation set visualization
        :param compute_metrics: bool, optional
                    If model metrics (accuracy, etc) should be computed. Default True.
        :param save_predictions:  bool, optional
                    If the model's predictions (forecasts) should be saved to file. Default False.
        :param cross_validation: bool, optional. Default False.
                    Determines if different behaviour should be undertaken due to cross validation.
        :return: dict
                 dictionary of metrics
        """
        raise NotImplementedError('evaluate must be implemented by a subclass')

    def make_model(self, key, **kwargs):
        """
        Builds a model.

        Method must be overridden by Experiment subclass.

        :param key: string
                    A string to select which model architecture to use.
        :return: tf.keras.models.Model
        """
        raise NotImplementedError('make_model must be implemented by a subclass')

    def set_targets(self):
        """
        Defines whatever the current model will try to predict.

        Method must be overridden by Experiment subclass.
        """
        raise NotImplementedError('set_targets must be implemented by a subclass')

    def summary(self):
        """
        Gets the experiment's summary
        :return: dictionary, experiment summary.
        """
        return self._summary

    def find_season(self):
        """
        :return: String
            returns the region name based on the name of the config file passed into DatasetManager
        """
        if 'freeze' in self.dataset_manager.name.lower():
            return 'Freeze-up'
        elif 'breakup' in self.dataset_manager.name.lower():
            return 'Breakup'
        else:
            raise ValueError('Could not determine if the season is Freeze-up or Breakup from '
                             'configuration name')

    def find_region(self):
        """
        :return: String
            returns the region name based on the name of the config file passed into DatasetManager
            To add other regions, add identifying string from configuration name to if statements
        """
        if 'hudson' in self.dataset_manager.name.lower():
            return 'Hudson Bay'
        elif 'baffin' in self.dataset_manager.name.lower():
            return 'Baffin Bay'
        elif 'nwt' in self.dataset_manager.name.lower():
            return 'NWT'
        elif 'eastcoast' in self.dataset_manager.name.lower():
            return 'East Coast'
        ### Arctic region feature in progress - 2021/07/14
        elif 'arctic' in self.dataset_manager.name.lower():
            if 'panarctic' in self.dataset_manager.name.lower():
                return 'Pan Arctic'
            else:
                return 'Arctic'
        else:
            raise ValueError('Could not determine if the region name from configuration name')

    def run(self, n_runs, model_key, training_kwargs, model_kwargs, compute_metrics=True,
            save_predictions=False, **kwargs):

        """
        Run the experiment. Will train n_runs models of the type defined by model_key and model_kwargs.
        Each model is evaluated after training and, at the end, the best model's forecasts will be visualized.
        Best model being defined as the model with the highest validation set accuracy.

        If any errors are encountered during training or evaluation, it will be re-attempted up to 5 times before the
        overall routine will raise an Exception.

        :param n_runs: int
                The number of internal-repetitions of the experiment to ensure reproducibility
        :param make_model_function: Function
                A function which can be called which returns tuple (tf.keras.models.Model, architecture_name)
        :param training_kwargs: dictionary
                keywords to be passed to the training procedure
        :param model_key: string
                Must be the name of a valid model architecture. Valid architectures are defined in subclass.
        :param model_kwargs:
                keywords arguments to be passed into make_model

        :param save_predictions
                If predictions made by the trained model should be saved
        :param **kwargs: dictionary  of optional keyword arguments
                - visualization_level: int
                            Controls the level of forecast visualization to be applied to all trained models. Default 0.
                            0 -> No visualization
                            1 -> Test set visualization
                            2 -> Test and Val set visualization
                - visualize_best: int
                            Controls the level of forecast visualization to be applied to the best model. Default 1.
                            'Best' being defined as the highest validation set accuracy.
                            0 -> No visualization
                            1 -> Test set visualization
                            2 -> Test and Val set visualization
                - cross_validation: are we performing cross_validation
                Either normal or cross_validation
        :param model_dir: string
                Parent directory of one cross validation experiment
        :param fold_dir: string
                Directory used to save results of one cross validation fold
        :return: True if experiment completed without issues, False otherwise.
        """
        if not self.configured:
            raise ValueError('Experiment must be configured prior to run!')

        # validate inputs
        if None in self.inputs:
            raise NotImplementedError('Inputs should be specified with set_inputs')

        if None in self.targets:
            raise NotImplementedError('This class should be overridden, and targets  must be specified by subclass')

        if type(model_key) != str:
            raise TypeError('Received model_key of non-string type')

        if type(n_runs) != int:
            raise TypeError('Received n_runs of non-int type')
        if n_runs < 1:
            raise ValueError('Received invalid value of n_runs')

        if 'skip_evaluation' in kwargs:
            skip_evaluation = kwargs['skip_evaluation']
            if type(skip_evaluation) != bool:
                raise TypeError('Received kwarg \'skip_evaluation\' of non-bool type')
        else:
            skip_evaluation = False

        if 'visualization_level' in kwargs:
            visualize = kwargs['visualization_level']
            if type(visualize) != int:
                raise TypeError('Received kwarg \'visualization_level\' of non-int type')
            if visualize not in {0, 1, 2}:
                raise ValueError('Received kwarg \'visualization_level\' of invalid value. Must be 0, 1, or 2')
        else:
            visualize = 1

        if 'visualize_best' in kwargs:
            visualize_best = kwargs['visualize_best']
            if type(visualize_best) != int:
                raise TypeError('Received kwargs \' visualize_best \' of non-int type')
            if visualize not in {0, 1, 2}:
                raise ValueError('Received kwarg \'visualize_best\' of invalid value. Must be 0, 1, or 2')
        else:
            visualize_best = 1

        if 'cross_validation' in kwargs:
            cross_validation = kwargs['cross_validation']
            if type(cross_validation) != bool:
                raise TypeError('Received kwarg \'cross_validation\' of non-bool type')

            if 'model_dir' not in kwargs:
                raise NameError('Model directory not specified for cross validation')

            if 'fold_dir' not in kwargs:
                raise NameError('Fold directory not specified for cross validation')

            model_dir = kwargs['model_dir']
            fold_dir = kwargs['fold_dir']
        else:
            cross_validation = False
            model_dir = ''
            fold_dir = ''

        print('INFO: PROCESS PID = {}'.format(os.getpid()))

        self._summary = dict()

        self._summary['model_kwargs'] = model_kwargs
        self._summary['training_kwargs'] = training_kwargs
        self._summary['n_runs'] = n_runs

        architecture_name = model_key
        # done validating inputs
        all_model_names = []
        best = -1
        best_val_acc = 0

        for i in range(n_runs):
            local_summary = dict()
            train_kwargs = copy.deepcopy(training_kwargs)  # avoid corrupting dict
            clear_session()
            # Get the current UTC time, replace strings with underscores, and remove milliseconds.
            model_name = architecture_name + ("_{}".format(datetime.datetime.utcnow()).replace(" ", "-"))[0:-7]
            all_model_names.append(model_name)

            local_summary['name'] = model_name
            local_summary['val_years'] = self.val_years
            local_summary['test_years'] = self.test_years
            local_summary['training_kwargs'] = train_kwargs
            local_summary['model_kwargs'] = model_kwargs

            if cross_validation:
                savepath = os.path.join(self.current_path, model_dir, str(self.test_years[0]), fold_dir)
            else:
                savepath = os.path.join(self.current_path, model_name)

            if os.path.exists(savepath):
                raise ValueError('Model already exists?')
            os.makedirs(savepath)
            completed_without_crashing = False
            learned_well = False
            attempts = 0
            while not (completed_without_crashing and learned_well):
                try:
                    clear_session()

                    model = self.make_model(model_key, **model_kwargs)
                    history, model, train_time = self.train(model, savepath, model_name, **train_kwargs)

                    learned_well = True
                    for metric in self.minimum_performance_metrics.keys():
                        # these minimum performance metrics are used to ensure that the model did not diverge
                        # or otherwise fail to meet a pre-determined minimum level of performance on the training set.
                        threshold = self.minimum_performance_metrics[metric]
                        if history.history[metric][-1] < threshold:
                            learned_well = False
                            local_summary['training_time'] = train_time

                except Exception as e:
                    attempts = attempts + 1
                    if attempts == 5:
                        print("Attempted to train 5 times!")
                        cuda.close()
                        raise e
                    else:
                        clear_session()
                        print("WARNING: ENCOUNTERED AN ERROR BUT CONTINUING")
                        print(e)
                else:
                    completed_without_crashing = True

            if not skip_evaluation:
                if len(self.dataset_manager.val_years) > 0:
                    local_summary['training_history'] = \
                                            dict(train_loss=[float(x) for x in history.history['loss']],
                                                 val_loss=[float(x) for x in history.history['val_loss']],
                                                 train_acc=[float(x) for x in history.history['binary_accuracy']],
                                                 val_acc=[float(x) for x in history.history['val_binary_accuracy']])

                completed_without_crashing = False
                attempts = 0
                while not completed_without_crashing:
                    try:
                        metrics = self.evaluate(self.dataset_manager, model, model_name, savepath, visualize=visualize,
                                                compute_metrics=compute_metrics, save_predictions=save_predictions,
                                                cross_validation=cross_validation)
                    except Exception as e:
                        attempts = attempts + 1
                        if attempts == 5:
                            print("Attempted to evaluate 5 times!")
                            cuda.close()
                            raise e
                        else:
                            print("WARNING: ENCOUNTERED AN ERROR BUT CONTINUING")
                            print(e)
                    else:
                        completed_without_crashing = True

                if len(self.dataset_manager.val_years) > 0:
                    val_acc = metrics['val_metrics']['accuracy']['mean']
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best = i

                    local_summary['evaluation_metrics'] = metrics
                    try:
                        with open(os.path.join(savepath, 'summary.yaml'), 'w') as f:
                            yaml.safe_dump(local_summary, f)
                    except Exception as e:
                        print(traceback.format_exc())
                        print(e)
                        print('encountered non-critical error')

                    self._summary[str(i)] = local_summary

        if visualize_best > 0 and best != -1 and not skip_evaluation:
            best_model_name = all_model_names[best]
            savepath = os.path.join(self.current_path, best_model_name)
            clear_session()
            best_model = load_model(os.path.join(savepath, '{}.h5'.format(best_model_name)))
            print('Best model was {} with {} validation accuracy'.format(best_model_name, best_val_acc))
            # visualize but do not compute metrics to save time
            completed_without_crashing = False
            attempts = 0
            while not completed_without_crashing:
                try:
                    self.evaluate(self.dataset_manager, best_model, best_model_name, savepath, visualize=visualize_best,
                                  compute_metrics=compute_metrics, save_predictions=save_predictions,
                                  cross_validation=False)
                except Exception as e:
                    attempts = attempts + 1
                    if attempts == 5:
                        print("Attempted to visualize 5 times!")
                        cuda.close()
                        raise e
                    else:
                        print("WARNING: ENCOUNTERED AN ERROR BUT CONTINUING")
                        print(e)
                else:
                    completed_without_crashing = True

        del model
        del history
        gc.collect()

        return True

    def run_monthly(self, init_years, model_key, training_kwargs, model_kwargs, generate_graphs=True, **kwargs):
        """
        Training a mounthly model.

        Plots that are created include:
            - overall model performance at different lead days compared to climate normal performance
              at specific lead days
            - monthly accuracy plots, montly accuracy maps
            - freeze-up/breakup date correlation plots, 7 day accuracy plots

        :param init_years: int
                How many years to use for initial training
        :param model_key: string
                The name of the model architecture to be used
        :param training_kwargs: dictionary
                keywords to be passed to the training procedure
        :param model_kwargs: dictionary
                keywords arguments to be passed into make_model
        :param generate_graphs: bool
                if True, plots will be generated
        :param **kwargs: dictionary  of optional keyword arguments
                - visualization_level: int
                            Controls the level of forecast visualization to be applied to all trained models. Default 0.
                            0 -> No visualization
                            1 -> Test set visualization
                            2 -> Test and Val set visualization
                - visualize_best: bool
                            Controls the level of forecast visualization to be applied to the best model. Default 1.
                            'Best' being defined as the highest validation set accuracy.
                            0 -> No visualization
                            1 -> Test set visualization
                            2 -> Test and Val set visualization
        :return: True if experiment completed without issues, False otherwise.
        """

        if self.dataset_manager is None:
            raise ValueError('DatasetManager must be initialized prior to run!')

        if 'use_checkpoints' not in training_kwargs or training_kwargs['use_checkpoints']:
            print("Warning: use_checkpoints cannot be True. Changing to False")
            training_kwargs['use_checkpoints'] = False

        if 'use_early_stopping' not in training_kwargs or not training_kwargs['use_early_stopping']:
            print("Warning: use_early_stopping cannot be False. Changing to True")
            training_kwargs['use_early_stopping'] = True

        if 'monitor' not in training_kwargs:
            training_kwargs['monitor'] = 'binary_accuracy'

        years = self.dataset_manager.years

        forecast_days = self.dataset_manager.forecast_days_forward
        historic_days = self.dataset_manager.days_of_historic_input

        model_directory = model_key + ("_cross_val_res_{}".format(datetime.datetime.utcnow()).replace(" ", "-"))[0:-7]

        test_record = dict()
        folds = 1
        self.train_years = years[:init_years]
        self.val_years = [years[init_years]]
        self.test_years = years[init_years+1:init_years+2]
        test_record[0] = self.test_years
        savepath = os.path.join(self.current_path, model_directory, str(self.test_years[0]))

        self.dataset_manager.config(days_of_historic_input=historic_days, forecast_days_forward=forecast_days,
                                    validation_years=self.val_years, test_years=self.test_years,train_remainder=False,
                                    custom_train_years=self.train_years)

        fold_directory = "fold_0"
        kwargs['skip_evaluation'] = False
        kwargs['cross_validation'] = True
        kwargs['fold_dir'] = fold_directory
        kwargs['model_dir'] = model_directory
        kwargs['visualize_best'] = 0
        kwargs['visualization_level'] = 0
        self.run(1, model_key, training_kwargs, model_kwargs,
                 compute_metrics=True, save_predictions=True, **kwargs)

        clear_session()
        # save cross validation configurations into yaml file
        test_record["yearly_start"] = self.dataset_manager.start_yearly
        test_record["yearly_end"] = self.dataset_manager.end_yearly
        test_record["forecast_days_forward"] = self.dataset_manager.forecast_days_forward
        test_record["days_of_historic_input"] = self.dataset_manager.days_of_historic_input
        test_record["data_path"] = self.dataset_manager.data_path
        test_record["lat_bounds"] = self.dataset_manager.lat_bounds
        test_record["long_bounds"] = self.dataset_manager.long_bounds
        test_record["raster_size"] = self.dataset_manager.resolution

        with open(os.path.join(savepath, 'test_record.yaml'), 'w') as f:
            yaml.safe_dump(test_record, f)

        savepath = os.path.join(self.current_path, model_directory, str(self.test_years[0]))

        # combines kfold experiment into 1 cube
        gc.collect()
        ytrue = create_unified_kfold_array(savepath, folds, "test-data.npy",
                                           historic_days, forecast_days, np.uint8)
        ypredict = create_unified_kfold_array(savepath, folds, "test_preds.npy",
                                              historic_days, forecast_days, np.float16)
        normal = create_unified_kfold_array(savepath, folds, "test_climate_normals.npy",
                                            historic_days, forecast_days, np.uint8)
        persistence = create_unified_kfold_array(savepath, folds, "test_persistence.npy",
                                                 historic_days, forecast_days, np.uint8)

        # combines date file into one data cube
        aggregate_dates(savepath, folds)

        test_metrics, test_persist_metrics, test_normal_metrics = \
            calculate_kfold_metrics(self.dataset_manager, savepath, model_key)

        # summarize and save experiment results into yaml file
        local_summary = dict()

        local_summary['model_kwargs'] = model_kwargs
        local_summary['training_kwargs'] = training_kwargs
        local_summary['folds'] = folds

        local_summary['test_metrics'] = metrics_to_yaml_format(test_metrics)
        local_summary['test_persist_metrics'] = metrics_to_yaml_format(test_persist_metrics)
        local_summary['test_normal_metrics'] = metrics_to_yaml_format(test_normal_metrics)

        with open(os.path.join(savepath, 'cross_val_summary.yaml'), 'w') as f:
            yaml.safe_dump(local_summary, f)

        # create plots
        if generate_graphs:
            landmask = self.dataset_manager.raw_data[0, :, :, self.dataset_manager.find_landmask_channel()]
            dates = get_dates_from_csv(savepath)
            region = self.find_region()
            # plot monthly accuracy plots and maps
            lat_bounds = self.dataset_manager.lat_bounds
            long_bounds = self.dataset_manager.long_bounds

            # plot correlation map and 7 day accuracy plot of predicted vs observed freeze-up/breakup date at region
            # i.e. Hudson Bay, NWT


            outpath = os.path.join(savepath, 'evaluations', 'monthly_accuracy_maps')
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            # plot monthly accuracy maps
            per_month_accuracy_maps(ytrue=ytrue, ypredict=ypredict, forecast_dates=dates, mask=landmask, savepath=outpath,
                                    region_name=region, climate_norm=normal, lat_bounds=lat_bounds, lon_bounds=long_bounds)

            outpath = os.path.join(savepath, 'evaluations', 'monthly_accuracy_plots')
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            # plot monthly accuracy plots
            per_month_accuracy_plots(ytrue=ytrue, ypredict=ypredict, forecast_dates=dates, mask=landmask, savepath=outpath,
                                     region_name=region, climate_norm=normal, persistence=persistence)

    def retrain_model(self, model_key, init_years, n_runs, training_kwargs, model_kwargs, **kwargs):

        """
        Run the retraining experiment. Will retrain n_runs models from the path defined by model_key and model_kwargs.
        Each model is evaluated after training and, at the end, the best model's forecasts will be visualized.
        Best model being defined as the model with the highest validation set accuracy.

        If any errors are encountered during training or evaluation, it will be re-attempted up to 5 times before the
        overall routine will raise an Exception.

        :param model_key: string
                Must be the name of a valid model architecture. Valid architectures are defined in subclass.
        :param init_years: training year
        :param n_runs: int
                The number of internal-repetitions of the experiment to ensure reproducibility
        :param training_kwargs: dictionary
                keywords to be passed to the training procedure
        :param model_kwargs:
                keywords arguments to be passed into make_model
        :param **kwargs: dictionary  of optional keyword arguments
                - visualization_level: int
                            Controls the level of forecast visualization to be applied to all trained models. Default 0.
                            0 -> No visualization
                            1 -> Test set visualization
                            2 -> Test and Val set visualization
                - visualize_best: int
                            Controls the level of forecast visualization to be applied to the best model. Default 1.
                            'Best' being defined as the highest validation set accuracy.
                            0 -> No visualization
                            1 -> Test set visualization
                            2 -> Test and Val set visualization
                - cross_validation: are we performing cross_validation
                Either normal or cross_validation
        :return: True if experiment completed without issues, False otherwise.
        """
        # if not self.configured:
        #     raise ValueError('Experiment must be configured prior to run!')

        # validate inputs
        if None in self.inputs:
            raise NotImplementedError('Inputs should be specified with set_inputs')

        if None in self.targets:
            raise NotImplementedError('This class should be overridden, and targets  must be specified by subclass')

        if type(n_runs) != int:
            raise TypeError('Received n_runs of non-int type')
        if n_runs < 1:
            raise ValueError('Received invalid value of n_runs')

        if 'skip_evaluation' in kwargs:
            skip_evaluation = kwargs['skip_evaluation']
            if type(skip_evaluation) != bool:
                raise TypeError('Received kwarg \'skip_evaluation\' of non-bool type')
        else:
            skip_evaluation = False

        if 'visualization_level' in kwargs:
            visualize = kwargs['visualization_level']
            if type(visualize) != int:
                raise TypeError('Received kwarg \'visualization_level\' of non-int type')
            if visualize not in {0, 1, 2}:
                raise ValueError('Received kwarg \'visualization_level\' of invalid value. Must be 0, 1, or 2')
        else:
            visualize = 1

        if 'visualize_best' in kwargs:
            visualize_best = kwargs['visualize_best']
            if type(visualize_best) != int:
                raise TypeError('Received kwargs \' visualize_best \' of non-int type')
            if visualize not in {0, 1, 2}:
                raise ValueError('Received kwarg \'visualize_best\' of invalid value. Must be 0, 1, or 2')
        else:
            visualize_best = 1

        if 'cross_validation' in kwargs:
            cross_validation = kwargs['cross_validation']
            if type(cross_validation) != bool:
                raise TypeError('Received kwarg \'cross_validation\' of non-bool type')

            if 'model_dir' not in kwargs:
                raise NameError('Model directory not specified for cross validation')

            if 'fold_dir' not in kwargs:
                raise NameError('Fold directory not specified for cross validation')

            model_dir = kwargs['model_dir']
            fold_dir = kwargs['fold_dir']
        else:
            cross_validation = False
            model_dir = ''
            fold_dir = ''

        if 'compute_metrics' in kwargs:
            compute_metrics = kwargs['compute_metrics']
            if type(compute_metrics) != bool:
                raise TypeError('Received kwarg \'compute_metrics\' of non-bool type')
        else:
            compute_metrics = False

        if 'save_predictions' in kwargs:
            save_predictions = kwargs['save_predictions']
            if type(save_predictions) != bool:
                raise TypeError('Received kwarg \'save_predictions\' of non-bool type')
        else:
            save_predictions = False

        generate_graphs = True

        years = self.dataset_manager.years

        forecast_days = self.dataset_manager.forecast_days_forward
        historic_days = self.dataset_manager.days_of_historic_input

        test_record = dict()
        folds = 1
        self.train_years = [years[init_years]]
        self.val_years = [years[init_years+1]]
        self.test_years = [years[init_years+2]]
        test_record[0] = self.test_years

        self.dataset_manager.config(days_of_historic_input=historic_days, forecast_days_forward=forecast_days,
                                    validation_years=self.val_years, test_years=self.test_years,train_remainder=False,
                                    custom_train_years=self.train_years)
        
        forecast_channels = list(np.arange(8))#[1]#[2, 4, 5]
#         forecast_channels.remove(self.dataset_manager.find_landmask_channel()) # remove landmask from forecast channels
         
        # create a copy of the experiment configuration for climate normal creation
        dsm2 = copy.deepcopy(self.dataset_manager)
        validation_years = self.val_years
        test_years = self.test_years 
        train_years = dsm2.years[:init_years+1]
        
        #moving climate normal as the last 10 years for dsm2 (rolling climate normal for augmented model input)
#         train_years = dsm2.years[init_years-9:init_years+1]
        

        if model_key == 'spatial_feature_pyramid_net_hiddenstate_ND':
            self.set_inputs([historical_all_channels(self.dataset_manager)])
        elif model_key == 'spatial_feature_pyramid_anomaly':
            self.set_inputs([historical_all_channels(self.dataset_manager),
                            historical_multi_channel_climate(self.dataset_manager, 
                                                             channels=forecast_channels, option='train', num_days=3)])
        elif model_key == 'spatial_feature_pyramid_hidden_ND_fc':
            dsm2.config(historic_days, forecast_days, validation_years,test_years,train_remainder=False,
                    custom_train_years=train_years)
            self.set_inputs([historical_all_channels(self.dataset_manager), 
                            future_multi_channel_climate(dsm2, forecast_channels, num_days = 90)])
        elif model_key == 'spatial_feature_pyramid_anomaly_fc':
            dsm2.config(historic_days, forecast_days, validation_years,test_years,train_remainder=False,
                    custom_train_years=train_years)
            self.set_inputs([historical_all_channels(self.dataset_manager),
                            historical_multi_channel_climate(self.dataset_manager,
                                                             channels=forecast_channels, option='train', num_days=3),
                            future_multi_channel_climate(dsm2, forecast_channels, num_days = 90)])
        else:
            raise NameError('Received invalid model name')
            
#         self.set_inputs([historical_all_channels(self.dataset_manager), future_multi_channel_climate(dsm2, 
#                                                                                        forecast_channels, 
#                                                                                        num_days = 90)])

        print('INFO: PROCESS PID = {}'.format(os.getpid()))

        self._summary = dict()

        self._summary['model_kwargs'] = model_kwargs
        self._summary['training_kwargs'] = training_kwargs
        self._summary['n_runs'] = n_runs

        architecture_name = model_key
        # done validating inputs
        all_model_names = []
        best = -1
        best_val_acc = 0

        for i in range(n_runs):
            local_summary = dict()
            train_kwargs = copy.deepcopy(training_kwargs)  # avoid corrupting dict
            clear_session()
            # Get the current UTC time, replace strings with underscores, and remove milliseconds.
            model_name = model_key + ("_cross_val_res_{}".format(datetime.datetime.utcnow()).replace(" ", "-"))[0:-7]
            all_model_names.append(model_name)

            local_summary['name'] = model_name
            local_summary['val_years'] = self.val_years
            local_summary['test_years'] = self.test_years
            local_summary['training_kwargs'] = train_kwargs
            local_summary['model_kwargs'] = model_kwargs

            # if cross_validation:
            #     savepath = os.path.join(self.current_path, model_dir, fold_dir)
            # else:
            #     savepath = os.path.join(self.current_path, model_name)
            savepath = os.path.join(self.current_path, model_dir, str(self.test_years[0]), fold_dir)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            completed_without_crashing = False
            learned_well = False
            attempts = 0
            while not (completed_without_crashing and learned_well):
                try:
                    clear_session()

                    # model_path = os.path.join(self.current_path,model_dir,fold_dir,model_dir)
                    path = os.path.join(self.current_path, model_dir, str(self.test_years[0]-1), fold_dir)
                    model_path = glob.glob(path + '/*.h5') #find saved model from previous year
                    model = self.make_model(model_key, **model_kwargs) #version change might have happened

                    print(model_path)
                    model.load_weights(model_path[0])

                    history, model, train_time = self.train(model, savepath, model_name, **train_kwargs)

                    learned_well = True
                    local_summary['training_time'] = train_time
                    for metric in self.minimum_performance_metrics.keys():
                        # these minimum performance metrics are used to ensure that the model did not diverge
                        # or otherwise fail to meet a pre-determined minimum level of performance on the training set.
                        threshold = self.minimum_performance_metrics[metric]
                        if history.history[metric][-1] < threshold:
                            learned_well = False

                except Exception as e:
                    attempts = attempts + 1
                    if attempts == 5:
                        print("Attempted to train 5 times!")
                        cuda.close()
                        raise e
                    else:
                        clear_session()
                        print("WARNING: ENCOUNTERED AN ERROR BUT CONTINUING")
                        print(e)
                else:
                    completed_without_crashing = True

            if not skip_evaluation:
                if len(self.dataset_manager.val_years) > 0:
                    local_summary['training_history'] = \
                                            dict(train_loss=[float(x) for x in history.history['loss']],
                                                 val_loss=[float(x) for x in history.history['val_loss']],
                                                 train_acc=[float(x) for x in history.history['binary_accuracy']],
                                                 val_acc=[float(x) for x in history.history['val_binary_accuracy']])

                completed_without_crashing = False
                attempts = 0
                while not completed_without_crashing:
                    try:
                        metrics = self.evaluate(dsm2, model, model_name, savepath, visualize=visualize,
                                                compute_metrics=compute_metrics, save_predictions=save_predictions,
                                                cross_validation=cross_validation)
                    except Exception as e:
                        attempts = attempts + 1
                        if attempts == 5:
                            print("Attempted to evaluate 5 times!")
                            cuda.close()
                            raise e
                        else:
                            print("WARNING: ENCOUNTERED AN ERROR BUT CONTINUING")
                            print(e)
                    else:
                        completed_without_crashing = True

                if len(self.dataset_manager.val_years) > 0:
                    val_acc = metrics['val_metrics']['accuracy']['mean']
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best = i

                    local_summary['evaluation_metrics'] = metrics
                    try:
                        with open(os.path.join(savepath, 'summary.yaml'), 'w') as f:
                            yaml.safe_dump(local_summary, f)
                    except Exception as e:
                        print(traceback.format_exc())
                        print(e)
                        print('encountered non-critical error')

                    self._summary[str(i)] = local_summary

        if visualize_best > 0 and best != -1 and not skip_evaluation:
            best_model_name = all_model_names[best]
            savepath = os.path.join(self.current_path, best_model_name)
            clear_session()
            best_model = load_model(os.path.join(savepath, '{}.h5'.format(best_model_name)))
            print('Best model was {} with {} validation accuracy'.format(best_model_name, best_val_acc))
            # visualize but do not compute metrics to save time
            completed_without_crashing = False
            attempts = 0
            while not completed_without_crashing:
                try:
                    self.evaluate(dsm2, best_model, best_model_name, savepath, visualize=visualize_best,
                                  compute_metrics=compute_metrics, save_predictions=save_predictions,
                                  cross_validation=False)
                except Exception as e:
                    attempts = attempts + 1
                    if attempts == 5:
                        print("Attempted to visualize 5 times!")
                        cuda.close()
                        raise e
                    else:
                        print("WARNING: ENCOUNTERED AN ERROR BUT CONTINUING")
                        print(e)
                else:
                    completed_without_crashing = True

        clear_session()
        # save cross validation configurations into yaml file
        test_record["yearly_start"] = self.dataset_manager.start_yearly
        test_record["yearly_end"] = self.dataset_manager.end_yearly
        test_record["forecast_days_forward"] = self.dataset_manager.forecast_days_forward
        test_record["days_of_historic_input"] = self.dataset_manager.days_of_historic_input
        test_record["data_path"] = self.dataset_manager.data_path
        test_record["lat_bounds"] = self.dataset_manager.lat_bounds
        test_record["long_bounds"] = self.dataset_manager.long_bounds
        test_record["raster_size"] = self.dataset_manager.resolution

        savepath = os.path.join(self.current_path, model_dir, str(self.test_years[0]))
        with open(os.path.join(savepath, 'test_record.yaml'), 'w') as f:
            yaml.safe_dump(test_record, f)

        # combines kfold experiment into 1 cube
        gc.collect()
        ytrue = create_unified_kfold_array(savepath, folds, "test-data.npy",
                                           historic_days, forecast_days, np.uint8)
        ypredict = create_unified_kfold_array(savepath, folds, "test_preds.npy",
                                              historic_days, forecast_days, np.float16)
        normal = create_unified_kfold_array(savepath, folds, "test_climate_normals.npy",
                                            historic_days, forecast_days, np.uint8)
        persistence = create_unified_kfold_array(savepath, folds, "test_persistence.npy",
                                                 historic_days, forecast_days, np.uint8)

        # combines date file into one data cube
        aggregate_dates(savepath, folds)

        test_metrics, test_persist_metrics, test_normal_metrics = \
            calculate_kfold_metrics(self.dataset_manager, savepath, model_key)

        # summarize and save experiment results into yaml file
        local_summary = dict()

        local_summary['model_kwargs'] = model_kwargs
        local_summary['training_kwargs'] = training_kwargs
        local_summary['folds'] = folds

        local_summary['test_metrics'] = metrics_to_yaml_format(test_metrics)
        local_summary['test_persist_metrics'] = metrics_to_yaml_format(test_persist_metrics)
        local_summary['test_normal_metrics'] = metrics_to_yaml_format(test_normal_metrics)

        with open(os.path.join(savepath, 'cross_val_summary.yaml'), 'w') as f:
            yaml.safe_dump(local_summary, f)

        # create plots
        if generate_graphs:
            landmask = self.dataset_manager.raw_data[0, :, :, self.dataset_manager.find_landmask_channel()]
            dates = get_dates_from_csv(savepath)
            region = self.find_region()
            # plot monthly accuracy plots and maps
            lat_bounds = self.dataset_manager.lat_bounds
            long_bounds = self.dataset_manager.long_bounds

            # plot correlation map and 7 day accuracy plot of predicted vs observed freeze-up/breakup date at region
            # i.e. Hudson Bay, NWT

            outpath = os.path.join(savepath, 'evaluations', 'monthly_accuracy_maps')
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            # plot monthly accuracy maps
            per_month_accuracy_maps(ytrue=ytrue, ypredict=ypredict, forecast_dates=dates, mask=landmask,
                                    savepath=outpath,
                                    region_name=region, climate_norm=normal, lat_bounds=lat_bounds,
                                    lon_bounds=long_bounds)

            outpath = os.path.join(savepath, 'evaluations', 'monthly_accuracy_plots')
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            # plot monthly accuracy plots
            per_month_accuracy_plots(ytrue=ytrue, ypredict=ypredict, forecast_dates=dates, mask=landmask,
                                     savepath=outpath,
                                     region_name=region, climate_norm=normal, persistence=persistence)

        del model
        del history
        gc.collect()

        return True