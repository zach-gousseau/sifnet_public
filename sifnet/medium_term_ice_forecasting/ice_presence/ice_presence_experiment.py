import os
from sifnet.experiment import Experiment
from sifnet.medium_term_ice_forecasting.ice_presence.ice_presence_evaluation import evaluation_procedure
from sifnet.data.GeneratorFunctions import future_single_channel_thresholded
from sifnet.medium_term_ice_forecasting.ice_presence.model import leaky_baseline_30_day_forecast, \
    baseline_30_day_forecast, spatial_feature_pyramid_net_vectorized_ND, spatial_feature_pyramid_net_hiddenstate_ND, \
    forecast_clstsm, spatial_feature_pyramid_anomaly


class IcePresenceExperiment(Experiment):

    def __init__(self,base_path='./results'):
        Experiment.__init__(self,base_path)
        self.name = 'IcePresenceExperiment'
        self.base_path = os.path.join(self.base_path, 'IcePresence')
        self.val_years = [1989, 2000, 2002, 2014, 2016]  # these can be manually overridden in configure if desired
        self.test_years = [1990, 2003, 2012, 2013, 2017]  # these can be manually overridden in configure if desired
        self.minimum_performance_metrics = {'binary_accuracy': 0.85, 'precision': 0.7, 'recall': 0.7}

    def set_targets(self):
        """
        sets the output/target type for the current experiment
        This is Ice presence, so we use future_single_channel_thresholded.
        If you wanted to use a different output/target type, you're using the wrong class.
        """
        if not self.configured:
            raise NotImplementedError('Cannot set targets if not configured.')

        self.targets = [future_single_channel_thresholded(self.dataset_manager)]

    def evaluate(self, dsm2, model, model_name, savepath, visualize, compute_metrics=True, save_predictions=False,
                 cross_validation=False):
        """
        :param model: tf.keras.models.Model object
        :param model_name: string
        :param savepath: string (path-like)
        :param visualize: int
                    0 -> no visualization
                    1 -> test set visualization
                    2 -> test and validation set visualization
        :param compute_metrics: bool, optional
                    If model metrics (accuracy, etc) should be computed. Default True.
        :param save_predictions: bool, optional
                    If predictions should be saved
        :param cross_validation: bool
                    If this function is used for cross validation
        :return: dict
                 dictionary of metrics
        """

        return evaluation_procedure(self.dataset_manager, dsm2, model, model_name, self.inputs, self.targets,
                                    savepath, self.current_path, visualize, restrict_to_gpu=False,
                                    compute_metrics=compute_metrics, save_predictions=save_predictions,
                                    cross_validation=cross_validation)

    def make_model(self, key, **kwargs):
        """
        :param key: string
            Specifies which model you want to use. Models ending with fc include future channels, and will
            take 2 inputs. The suggested model without future channel is spatial_feature_pyramid_net_hiddenstate_ND. The
            suggested model with future channels is spatial_feature_pyramid_hidden_ND_fc or SFP_HS_LC_ND_fc.
        :param kwargs: parameters to pass into the model
        :return: dict
                 dictionary of metrics
        """
        model_dict = dict(leaky_baseline_30D=leaky_baseline_30_day_forecast,
                          baseline_30D=baseline_30_day_forecast,
                          forecast_clstsm=forecast_clstsm,
                          spatial_feature_pyramid_net_vectorized_ND=spatial_feature_pyramid_net_vectorized_ND,
                          spatial_feature_pyramid_net_hiddenstate_ND=spatial_feature_pyramid_net_hiddenstate_ND,
                          spatial_feature_pyramid_anomaly=spatial_feature_pyramid_anomaly)
        if key in model_dict:
            return model_dict[key](**kwargs)
        else:
            raise ValueError('model_key not found. Received {}, valid values are {}'.format(key, model_dict.keys()))
