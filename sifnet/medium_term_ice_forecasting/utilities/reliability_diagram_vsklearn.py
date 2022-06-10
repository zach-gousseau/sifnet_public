import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from matplotlib.ticker import PercentFormatter


def reliability_diagram(ytrue, yconf, landmask, save_path, title, num_bins = 10, plot_strategy = 'uniform'):
    """
    Plots Reliability Diagrams.

        You cannot use traditional evaluation methods for this model because our output is a float [0,1] and our ytrue
        is an explicit boolean of 1 or 0. Instead we use a reliability diagram. The goal of a reliability diagram is
        to compare our float number of forecasted probability to the frequency that the event 'ice presence' occurs.
        For a perfect forecast, these numbers should be equal for all values of forecast probability.

        For more information on how sklearn's implementation of a reliability diagram, check:
            https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html

        *Note: reliability diagram and calibration curves are equivalent.

        :param ytrue: numpy.ndarray (4D). Contains array of shape [num_days, days_forecasted_forwards, x, y]
            with 1 (ice) or 0 (no ice).
        :param yconf: numpy.ndarray (4D). Output of SIF-Net; contains array of shape
            [num_days, days_forecasted_forwards, x, y] with float for probability of ice presence.
        :param landmask: numpy.ndarray (2D). Contains array of shape [x, y] with 1 (land) or 0 (no land).
        :param save_path: string. Location where resulting reliability diagram and distribution graphs are saved.
        :param title: string. Title for each graph. 'Reliability Diagram' or 'Distribution Graph'
            is inserted into title.
        :param num_bins: int. Number of bins used for x-axis of reliability diagrams.
        :param plot_strategy: string. Valid values are 'uniform', 'quantile'.
            uniform -> All bins have identical widths.
            quantile -> All bins have the same number of points.
            Uniform is used by default. Since model outputs crowd around 1 and 0, when using quantile binning,
            plotting points also crowd around 1 and 0. This causes a majority of the graph to be interpolated,
            defeating the purpose of the reliability diagram.
        :return: None
    """

    # checking that inputs are valid
    if type(ytrue) != np.ndarray:
        raise TypeError('ytrue must be of type numpy.ndarray')
    if len(ytrue.shape) != 4:
        raise ValueError('ytrue must be a 4D array')
    if type(yconf) != np.ndarray:
        raise TypeError('yconf must be of type numpy.ndarray')
    if len(yconf.shape) != 4:
        raise ValueError('yconf must be a 4D array')
    if ytrue.shape != yconf.shape:
        raise ValueError('shape of ytrue and yconf must be the same')
    if type(landmask) != np.ndarray:
        raise TypeError('landmask must be of type numpy.ndarray')
    if len(landmask.shape) != 2:
        raise ValueError('landmask must be a 2D array')
    if ytrue.shape[2] != landmask.shape[0] or ytrue.shape[3] != landmask.shape[1]:
        raise ValueError('landmask raster shape does not agree with shape of ytrue and yconf rasters')
    if type(save_path) != str:
        raise TypeError('save_path must be of type string')
    if not os.path.exists(save_path):
        raise IOError('save_path does not exist')
    if type(title) != str:
        raise TypeError('title must be of type string')
    if type(num_bins) != int:
        raise TypeError('num_bins must be of type int')
    if num_bins <= 0:
        raise ValueError('num_bins must be greater than 0')
    if type(plot_strategy) != str:
        raise TypeError('plot_strategy must be of type string')
    if plot_strategy != 'quantile' and plot_strategy != 'uniform':
        raise ValueError('plot_strategy must have value \'uniform\' or\'quantile\'')

    print("Forming Reliability Diagrams")

    # create arrays holding necessary data for 15(, 30, 60, 90) day plot(s)
    ytrues = []
    yconfs = []
    allowable_values = [15, 30, 60, 90]
    values_special = [31, 62, 93]

    # if allowable shape received by method
    if (ytrue.shape[1] in allowable_values or ytrue.shape[1] in values_special) and ytrue.shape == yconf.shape:
        # determining number of graphs based off of input data cube shape
        num_plots = 0
        if ytrue.shape[1] == 15:
            num_plots = 1
        elif ytrue.shape[1] == 30 or ytrue.shape[1] == 31:
            num_plots = 2
        elif ytrue.shape[1] == 60 or ytrue.shape[1] == 62:
            num_plots = 3
        elif ytrue.shape[1] == 90:
            num_plots = 4

        # squeezing landmask to 1D array
        landmask = landmask.reshape(landmask.shape[0] * landmask.shape[1])

        print('Computations at 0%')

        # for number of graphs
        for _ in range(int(num_plots)):

            # accessing lead day, squeezing to 1D array, creating list
            ytrue_ = ytrue[:, allowable_values[_] - 1, :, :]
            ytrue_ = ytrue_.reshape(ytrue_.shape[0] * 1 * ytrue_.shape[1] * ytrue_.shape[2])
            ytrue_ = list(ytrue_)

            yconf_ = yconf[:, allowable_values[_] - 1, :, :]
            yconf_ = yconf_.reshape(yconf_.shape[0] * 1 * yconf_.shape[1] * yconf_.shape[2])
            yconf_ = list(yconf_)

            # removing land using landmask
            # if landmask at same point in 'x by y' raster is 0 (no land) keep the point in 1D array
            ytrue_temp = [ytrue_[i] for i in range(len(ytrue_)) if landmask[i % len(landmask)] == 0]
            yconf_temp = [yconf_[i] for i in range(len(yconf_)) if landmask[i % len(landmask)] == 0]

            # adding array for each lead day to master array
            ytrues.append(ytrue_temp)
            yconfs.append(yconf_temp)

            print('Computations at ' + str((_ + 1) / 4 * 100) + '%')

        print('Computations at 100%\n')

        # master arrays for plotting; contain sub-arrays for each lead day
        prob_trues = []
        prob_preds = []

        # for each lead day
        for i in range(num_plots):
            # create array with points for reliability diagram, append to master array
            prob_true, prob_pred = calibration_curve(y_true = ytrues[i], y_prob = yconfs[i], n_bins = num_bins,
                                                     strategy = plot_strategy)
            prob_trues.append(prob_true)
            prob_preds.append(prob_pred)

        # plotting reliability diagrams
        perf_line = [0, 1.0]
        plt.plot(perf_line, perf_line, linestyle = 'dashed', label = 'Perfect Forecast')
        plt.xlabel('Forecast Probability')
        plt.ylabel('Observed Frequency')
        title_name = 'Reliability Diagram ' + title
        plt.title(title_name)
        # for each lead day
        for i in range(num_plots):
            # plotting each array in master array on same graph
            plt.plot(prob_preds[i], prob_trues[i], label = str(allowable_values[i]) + ' Lead Day', marker = '*')
        plt.legend()
        save_name = 'Reliability_Diagram_' + title.replace(' ', '_')
        save_file_name = os.path.join(save_path, save_name)
        plt.savefig(fname = save_file_name)
        print("Reliability Diagram saved at " + str(save_file_name))
        plt.show()

        # plotting distribution graphs
        title_name = 'Distribution Graph ' + title
        plt.title(title_name)
        plt.xlabel('Forecast Probability')
        plt.ylabel('Distribution')

        # plotting histogram that shows the distribution of input data
        plt.hist(x = yconfs, bins = 10, label = ['15 Lead Day', '30 Lead Day', '60 Lead Day', '90 Lead Day'])
        save_name = 'Distribution_Graph_' + title.replace(' ', '_')
        save_file_name = os.path.join(save_path, save_name)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(len(yconfs[0])))
        plt.tight_layout()
        plt.legend()
        plt.savefig(fname = save_file_name)
        print("Distribution Graph saved at " + str(save_file_name))
        plt.show()

    else:
        raise Exception("Something wrong with shapes of input data cubes...")

if __name__ == "__main__":
    print("Loading numpy files...")
    ytrue_main = np.load('/work/local_data/results/IcePresence/EastCoast/'
                         'EastCoast_Breakup_v2_reduced_extended_Geomet_forecast/H2-F62/'
                         'spatial_feature_pyramid_hidden_ND_fc_cross_val_res_2020-04-06-20:37:57/'
                         'aggregated_test-data.npy')
    yconf_main = np.load('/work/local_data/results//IcePresence/EastCoast/'
                         'EastCoast_Breakup_v2_reduced_extended_Geomet_forecast/H2-F62/'
                         'spatial_feature_pyramid_hidden_ND_fc_cross_val_res_2020-04-06-20:37:57/'
                         'aggregated_test_preds.npy')
    landmask_main = np.load('/work/local_data/EastCoast_Breakup_v2_reduced_extended_Geomet_forecast.npy')[0, :, :, 5]
    print("Done loading numpy files.")
    reliability_diagram(ytrue = ytrue_main, yconf = yconf_main, landmask = landmask_main,
                        save_path = '/work/local_data/results', title = 'test', num_bins = 10,
                        plot_strategy = 'uniform')
