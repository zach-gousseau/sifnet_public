"""
This file represents a collection of functions that may be used as building-blocks to create custom generators.
Each function takes as its first argument the DatasetManager object to which it will be attached.

The number of inputs provided by the generator will be equal to the number of input-functions which have been configured.
Similarly, the number of outputs will be equal to the number of output-functions which have been configured.
Addition 'auxiliary' outputs may also be provided. These should not be used for training a model, but can be useful.

Each function herein returns a tuple of length 3.
The returns are (function pointer, sample_shape, datatype)
 - function pointer is used internally by the generator
    > Each of these returned functions take a single argument; a numpy.ndarray with shape=(batch_size, sample_shape).
        >> More accurately, shape = (batch_size, sample_shape[0], ... sample_shape[n-1])
    > The returned function itself also returns a function pointer, a 'work function'
    > The work function tells the generator's worker-threads how to fill the numpy array with individual samples.
 - sample shape is the dimensions of one independent sample.
    > For example, [3, 160, 300, 8] from historical_all_channels represents [N days, latitude, longitude, n_channels]
 - datatype dictates the datatype of the numpy array that should be passed back into the returned function
    > Can be None to indicate default/unspecified
    > others might include {np.uint8, np.float16, datetime.date)


Notes
-----
On the subject of 'channels':
    A channel defines one individual variable which is present at each location.
    For example, it is easy to think of an RGB image as 2D with shape [Height, Width],
        but it is actually 3D with shape [Height, Width, Channels].
        In this case there would be 3 channels, corresponding to the red, green, and blue components of each pixel.

        Similarly, the channels found in our input data  at each location correspond to temperature, winds, etc.

    We use the TensorFlow convention of channels last, as opposed to the Theano convention of channels first.
        So for comparison, [Height, Width, Channels] instead of [Channels, Height, Width].
    Keras will use this same convention by default because we are importing it from TensorFlow,
    though it can be configured to use either.
"""

import numpy as np
from datetime import date
# from PIL import Image


def historical_all_channels(dataset):  # F1
    """
    A function for use with DatasetManager.make_generator
    Creates an input or output consisting of all available data channels, in the past
    Duration is days_of_historic_input as specified in config()
    :param dataset: DatasetManager object.
                    Source of raw data.
    :return: tuple (function pointer, sample_shape, None)
    """
    shape = dataset.raw_data[0].shape
    shape = [dataset.days_of_historic_input] + list(shape)  # e.g. [3, 160, 300, 8]

    def _historical_all_channels(arr):  # F2
        """
        _past_history_all_channels(dataset, n, arr) creates a function which recieves a single argument
        containing i,j indicies and uses them to place a single input into the placeholder array.
        The returned function may then be passed to ThreadExecutorPool or similar concurrent execution method.
        Note that returned datatype 'None' indicates to use the default datatype
        :param arr: Placeholder array np.ndarray
        :return: (function, shape, None)
        """
        n = dataset.days_of_historic_input

        def _phac(arg):  # F3
            """
            This is the internal work-function. Executed concurrently by N=batch_size threads.
            :param arg: tuple(int, int)
                        i - lookup-table index of current sample
                        j - worker-thread's ID.
            :return: 1 - indicates that the job is complete
            """
            i, j = arg
            arr[j] = dataset.raw_data[(i - n + 1):(i + 1)]
            return 1

        return _phac

    return _historical_all_channels, list(shape), None  # function, shape of output

def historical_multi_channel_climate(dataset, channels, option='train+val', num_days = 3):
    """
    A function for use with DatasetManager.make_generator
    Creates an input or output consisting of the specified data channels.
    Duration is forecast_days_forward
    Note that returned datatype 'None' indicates to use the default datatype
    :param dataset: DatasetManager object.
                    Source of raw data.
    :param channels: list of desired channels e.g [1,2,3,4,5]
    :param downscale_factor: optional, if provided then must be a power of 2.
                                The output will be of a fraction of the size
    :param constrain: boolean, optional. Default True
                     If the output should be constrained to be no more than 30 days (forecast input constraint)
    :param num_days: int, optional. Provides the number of days of forecasted information to produce
    :return: tuple(function pointer, sample_shape, None)
    """
    """
    A function for use with DatasetManager.make_generator
    To calculate the climate normal for a target date, it first finds all data from the selected subset
        subset being, training set, training & val sets, or traing & val & test sets with this same date.
    From the data, it applied the concentration threshold (default 0.15) and finds the percentage of
        all samples where this criteria is met.
    If more than the desired percentage (default 50%) of samples have met this criteria, returns 1. Otherwise 0.

    Note that returned datatype 'None' indicates to use the default datatype
    :param dataset: DatasetManager object.
                    Source of raw data.
    :param option: string
                    climate normal option. Base normal on training set, training+validation sets, or all data.
                    valid options are {'train', 'train+val', 'all'}
    :param channel: int
                    ID/index of channel of interest. Default zero (sea ice)
    :param threshold: Float, optional. Default 0.15
    :param prob_threshold: Float, optional. Default 0.5
    :return: function pointer, shape, None
    """
    if option not in {'train', 'train+val', 'all'}:
        raise ValueError("Climate normal func option must be one of {'train', 'train+val', 'all'}")

    if option == 'train':
        years = dataset.train_years
    elif option == 'train+val':
        years = sorted(np.concatenate((dataset.train_years, dataset.val_years)))
    elif option == 'all':
        years = dataset.years

    shape = list(dataset.raw_data[0].shape)[0:-1]
    n = num_days
    channels = np.array(channels, dtype=np.int)
    
    shape = [n] + list(shape) + [len(channels)]
    
    def _climate_normal(arr):
        """.
        :param arr: placeholder array np.ndarray
        :return: function pointer
        """
#         n = dataset.forecast_days_forward

        def _cln(arg):
            i, j = arg
            this_date = dataset.all_dates[i]
            month = this_date.month
            day = this_date.day
            leap_year = (month == 2 and day == 29)  # bool
            data = []
            relavant_dates = [d for d in dataset.all_dates if ((d.year in years)
                                                               and (d.month == month and d.day == day))]
            if leap_year:
                additional_dates = [date(year, 3, 1) for year in years if not dataset.is_leap_year(year)
                                    and (year != dataset.years[0] or dataset.start_yearly[0] < 3)]
                relavant_dates = relavant_dates + additional_dates

            for D in relavant_dates:
                index = dataset.find_index_of_date(D)
                data.append(dataset.raw_data[(index - n + 1):(index + 1), :, :, channels])

            data = np.stack(data, axis=0)
            data = np.mean(data, axis=0)

            arr[j] = data#np.expand_dims(data, axis=-1)
            return 1

        return _cln

    return _climate_normal, shape, None


def future_single_channel_thresholded(dataset, channel=0, threshold=0.15):
    """
    A function for use with DatasetManager.make_generator
    Creates an input or output consisting of the specified data channel, in the future, with a binary threshold as given
    The primary output function initially used for sea-ice-presence.
    Duration is forecast_days_forward
    Note that returned datatype 'None' indicates to use the default datatype
    :param dataset: DatasetManager object.
                    Source of raw data.
    :param channel: index of relevant channel
    :param threshold: binary threshold value
    :return: tuple(function pointer, sample_shape, None)
    """
    shape = list(dataset.raw_data[0].shape)[0:-1]
    shape = [dataset.forecast_days_forward] + shape + [1]  # e.g [30, SHAPE, 1] -> [30, 160, 300, 1]
    print(shape)

    def _future_single_channel_thresholded(arr):
        """
        :param arr: Placeholder array np.ndarray
        :return function pointer
        """
        n = dataset.forecast_days_forward

        def _fsct(arg):
            i, j = arg
            try:
                # creates a slice of the data array based on a single channel and applies a binary threshold
                # 'i' is the index of 'today'. So i+1 : i + n + 1 creates a slice including tomorrow and the next n days
                # channel = 0 normally indicates sea ice concentration, so tomorrow through day n's ice concentrations
                # expand_dims ensures that the there is an axis (size 1) relating to the single input channel.
                # necessary because TensorFlow & keras always expect a channels axis, even if it is singular in size.
                # and would normally be meaningless.
                arr[j] = np.expand_dims(dataset.raw_data[i + 1:i + n + 1, :, :, channel] > threshold, axis=-1)
            except ValueError:
                # was this function causing issues? Didn't notice when this was added.
                print(np.expand_dims(dataset.raw_data[i + 1:i + n + 1, :, :, channel] > threshold, axis=-1).shape)
                print(i, j)
                print()
                print(dataset.all_dates[i])
                raise Exception
            return 1

        return _fsct

    return _future_single_channel_thresholded, shape, None  # same shape but only one channel


def future_multi_channel(dataset, channels, downscale_factor=None, constrain=True, num_days = 30):
    """
    A function for use with DatasetManager.make_generator
    Creates an input or output consisting of the specified data channels.
    Duration is forecast_days_forward
    Note that returned datatype 'None' indicates to use the default datatype
    :param dataset: DatasetManager object.
                    Source of raw data.
    :param channels: list of desired channels e.g [1,2,3,4,5]
    :param downscale_factor: optional, if provided then must be a power of 2.
                                The output will be of a fraction of the size
    :param constrain: boolean, optional. Default True
                     If the output should be constrained to be no more than 30 days (forecast input constraint)
    :param num_days: int, optional. Provides the number of days of forecasted information to produce
    :return: tuple(function pointer, sample_shape, None)
    """
    shape = list(dataset.raw_data[0].shape)[0:-1]
    channels = np.array(channels, dtype=np.int)

#     if downscale_factor:
#         if not (downscale_factor != 0 and not (downscale_factor & downscale_factor - 1)):
#             # if not power of 2
#             raise ValueError("downscale_factor must be a power of 2")
#         shape = np.array(shape)
#         if not (np.all(np.mod(shape, downscale_factor) == 0)):
#             raise ValueError("downscale factor does not evenly divide shape")

#         shape = shape / downscale_factor  # (160, 300) -> (80, 150)
#         shape = np.array(shape, dtype=int)

#         local_array = np.ndarray(shape=([len(dataset.raw_data)] + list(shape) + [len(channels)]), dtype=np.float)

#         temp = dataset.raw_data[:, :, :, channels]  # extract relevant channels
#         for ind in range(len(local_array)):
#             for chan in range(len(channels)):
#                 sample = temp[ind, :, :, chan]
#                 sample = Image.fromarray(sample, mode='F')
#                 sample = sample.resize(np.flip(shape, 0), resample=Image.BICUBIC)  # Different resample?
#                 local_array[ind, :, :, chan] = np.array(sample)

#         del temp
#     else:
    local_array = dataset.raw_data[:, :, :, channels]

    if constrain:
        n = min(dataset.forecast_days_forward, num_days)
    else:
        n = num_days

    shape = [n] + list(shape) + [len(channels)]

    def _future_multi_channel(arr):
        """
        :param arr: Placeholder array np.ndarray
        :return: function pointer
        """

        def _fmc(arg):
            i, j = arg
            arr[j] = local_array[i + 1:i + n + 1]
            return 1

        return _fmc

    return _future_multi_channel, shape, None


def future_multi_channel_climate(dataset, channels, option='train+val', num_days = 30):
    """
    A function for use with DatasetManager.make_generator
    Creates an input or output consisting of the specified data channels.
    Duration is forecast_days_forward
    Note that returned datatype 'None' indicates to use the default datatype
    :param dataset: DatasetManager object.
                    Source of raw data.
    :param channels: list of desired channels e.g [1,2,3,4,5]
    :param downscale_factor: optional, if provided then must be a power of 2.
                                The output will be of a fraction of the size
    :param constrain: boolean, optional. Default True
                     If the output should be constrained to be no more than 30 days (forecast input constraint)
    :param num_days: int, optional. Provides the number of days of forecasted information to produce
    :return: tuple(function pointer, sample_shape, None)
    """
    """
    A function for use with DatasetManager.make_generator
    To calculate the climate normal for a target date, it first finds all data from the selected subset
        subset being, training set, training & val sets, or traing & val & test sets with this same date.
    From the data, it applied the concentration threshold (default 0.15) and finds the percentage of
        all samples where this criteria is met.
    If more than the desired percentage (default 50%) of samples have met this criteria, returns 1. Otherwise 0.

    Note that returned datatype 'None' indicates to use the default datatype
    :param dataset: DatasetManager object.
                    Source of raw data.
    :param option: string
                    climate normal option. Base normal on training set, training+validation sets, or all data.
                    valid options are {'train', 'train+val', 'all'}
    :param channel: int
                    ID/index of channel of interest. Default zero (sea ice)
    :param threshold: Float, optional. Default 0.15
    :param prob_threshold: Float, optional. Default 0.5
    :return: function pointer, shape, None
    """
    if option not in {'train', 'train+val', 'all'}:
        raise ValueError("Climate normal func option must be one of {'train', 'train+val', 'all'}")

    if option == 'train':
        years = dataset.train_years
    elif option == 'train+val':
        years = sorted(np.concatenate((dataset.train_years, dataset.val_years)))
    elif option == 'all':
        years = dataset.years

    shape = list(dataset.raw_data[0].shape)[0:-1]
    n = num_days
    channels = np.array(channels, dtype=np.int)
    
    shape = [n] + list(shape) + [len(channels)]
    
    def _climate_normal(arr):
        """.
        :param arr: placeholder array np.ndarray
        :return: function pointer
        """
#         n = dataset.forecast_days_forward

        def _cln(arg):
            i, j = arg
            this_date = dataset.all_dates[i]
            month = this_date.month
            day = this_date.day
            leap_year = (month == 2 and day == 29)  # bool
            data = []
            relavant_dates = [d for d in dataset.all_dates if ((d.year in years)
                                                               and (d.month == month and d.day == day))]
            if leap_year:
                additional_dates = [date(year, 3, 1) for year in years if not dataset.is_leap_year(year)
                                    and (year != dataset.years[0] or dataset.start_yearly[0] < 3)]
                relavant_dates = relavant_dates + additional_dates

            for D in relavant_dates:
                index = dataset.find_index_of_date(D)
                data.append(dataset.raw_data[index+1:index+n+1, :, :, channels])

            data = np.stack(data, axis=0)
            data = np.mean(data, axis=0)
            # if threshold:
            #     data = data > threshold
#             data = data > threshold  # for all years, check if concentration greater than threshold
#             data = np.mean(data, axis=0)  # convert to probability of concentration greater than threshold
#             data = data > prob_threshold  # apply probability threshold

            arr[j] = data#np.expand_dims(data, axis=-1)
            return 1

        return _cln

    return _climate_normal, shape, None


def ice_concentration_forecast(dataset, channel=0):
    """
    A function for use with DatasetManager.make_generator
    Creates an input or output consisting of the specified channel with binning as described below.
    Concentration bins are based on the World Meteorological Organization's colour code for total concentration
    Reference:
    https://www.canada.ca/en/environment-climate-change/services/ice-forecasts-observations
                                /publications/interpreting-charts/chapter-2.html#concentration-sea
                        ^Careful copying that URL, the line break introduces a space that you must remove.
    Code is as follows:
        - Ice free (0)
        - Open water: Less than one tenth
        - Very open ice: One tenth to three tenths
        - Open ice: Four tenths to six tenths
        - Close ice: Seven tenths to 8 Tenths
        - Very close ice: Nine tenths to 100%
    These bins must be adapted to continuous-range bins. The following adaptation is currently employed:
        -Ice free: [0]
        -Open water: (0, 0.1)
        -Very Open ice: [0.1, 0.35)
        -Open ice: [0.35, 0.65]
        -Close ice: (0.65, 0.85)
        -Very close ice: [0.85, 1.0]
    Note that round and square brackets are used above to denote non-inclusive and inclusive ranges respectively.
    Note that returned datatype 'None' indicates to use the default datatype
    :param dataset: DatasetManager object.
                    Source of raw data.
    :param channel: int
                    index/ID of desired data channel. Default zero for sea ice concentration
    :return: function pointer, shape, None
    """
    if dataset.vars[channel]['type'] != 'ci':
        print(dataset.vars[channel])
        raise ValueError("Incorrect channel specified or sea-ice variable does not exist.")

    shape = list(dataset.raw_data[0].shape)[0:-1]

    temp = dataset.raw_data[:, :, :, channel]
    local_array = np.ndarray(shape=(list(dataset.raw_data.shape[0:-1]) + [6]))  # 4892, 160, 300, 6

    local_array[:, :, :, 0] = (temp == 0).astype(np.uint8)  # Ice Free
    local_array[:, :, :, 1] = ((temp > 0) * (temp < 0.1)).astype(np.uint8)  # Open Water
    local_array[:, :, :, 2] = ((temp >= 0.1) * (temp < 0.35)).astype(np.uint8)  # Very open ice
    local_array[:, :, :, 3] = ((temp >= 0.35) * (temp <= 0.65)).astype(np.uint8)  # Open ice
    local_array[:, :, :, 4] = ((temp > 0.65) * (temp < 0.85)).astype(np.uint8)  # Close ice
    local_array[:, :, :, 5] = (temp >= 0.85).astype(np.uint8)  # Very close ice

    shape = [dataset.forecast_days_forward] + shape + [6]
    print(local_array.shape)
    def _conc(arr):
        """
        :param arr: placeholder array np.ndarray
        :return: function pointer
        """
        n = dataset.forecast_days_forward

        def _conc_f(arg):
            i, j = arg
            arr[j] = local_array[i+1: i+n+1]
            return 1

        return _conc_f

    return _conc, shape, None


def persistence(dataset, channel=0, threshold=None):
    """
    A function for use with DatasetManager.make_generator
    Creates an output consisting of the specified channel repeated for each forecast day
    Note that returned datatype 'None' indicates to use the default datatype
    :param dataset: DatasetManager object.
                    Source of raw data.
    :param channel: int
                    index/ID of desired data channel. Default zero for sea ice concentration.
    :param threshold: float, optional
                    If provided, outputs will be thresholded at the given value
                    e.g outputs = outputs > threshold
    :return: function pointer, shape, None
    """
    shape = list(dataset.raw_data[0].shape)[0:-1]
    shape = [dataset.forecast_days_forward] + shape + [1]

    def _persistence(arr):
        """
        :param arr: placeholder array np.ndarray
        :return: function pointer
        """
        n = dataset.forecast_days_forward

        def _pf(arg):
            i, j = arg
            temp = np.expand_dims(dataset.raw_data[i, :, :, channel], axis=-1)
            temp = np.expand_dims(temp, axis=0)
            temp = np.repeat(temp, n, axis=0)

            if threshold:
                temp = (temp > threshold).astype(np.uint8)

            arr[j] = temp
            return 1

        return _pf

    return _persistence, shape, None


def climate_normal(dataset, option='train+val', channel=0, threshold=0.15, prob_threshold=0.5):
    """
    A function for use with DatasetManager.make_generator
    To calculate the climate normal for a target date, it first finds all data from the selected subset
        subset being, training set, training & val sets, or traing & val & test sets with this same date.
    From the data, it applied the concentration threshold (default 0.15) and finds the percentage of
        all samples where this criteria is met.
    If more than the desired percentage (default 50%) of samples have met this criteria, returns 1. Otherwise 0.

    Note that returned datatype 'None' indicates to use the default datatype
    :param dataset: DatasetManager object.
                    Source of raw data.
    :param option: string
                    climate normal option. Base normal on training set, training+validation sets, or all data.
                    valid options are {'train', 'train+val', 'all'}
    :param channel: int
                    ID/index of channel of interest. Default zero (sea ice)
    :param threshold: Float, optional. Default 0.15
    :param prob_threshold: Float, optional. Default 0.5
    :return: function pointer, shape, None
    """
    if option not in {'train', 'train+val', 'all'}:
        raise ValueError("Climate normal func option must be one of {'train', 'train+val', 'all'}")

    if option == 'train':
        years = dataset.train_years
    elif option == 'train+val':
        years = sorted(np.concatenate((dataset.train_years, dataset.val_years)))
    elif option == 'all':
        years = dataset.years

    shape = list(dataset.raw_data[0].shape)[0:-1]
    shape = [dataset.forecast_days_forward] + shape + [1]

    def _climate_normal(arr):
        """.
        :param arr: placeholder array np.ndarray
        :return: function pointer
        """
        n = dataset.forecast_days_forward

        def _cln(arg):
            i, j = arg
            this_date = dataset.all_dates[i]
            month = this_date.month
            day = this_date.day
            leap_year = (month == 2 and day == 29)  # bool
            data = []
            relavant_dates = [d for d in dataset.all_dates if ((d.year in years)
                                                               and (d.month == month and d.day == day))]
            if leap_year:
                additional_dates = [date(year, 3, 1) for year in years if not dataset.is_leap_year(year)
                                    and (year != dataset.years[0] or dataset.start_yearly[0] < 3)]
                relavant_dates = relavant_dates + additional_dates

            for D in relavant_dates:
                index = dataset.find_index_of_date(D)
                data.append(dataset.raw_data[index+1:index+n+1, :, :, channel])

            data = np.stack(data, axis=0)
            # data = np.mean(data, axis=0)
            # if threshold:
            #     data = data > threshold
            data = data > threshold  # for all years, check if concentration greater than threshold
            data = np.mean(data, axis=0)  # convert to probability of concentration greater than threshold
            data = data > prob_threshold  # apply probability threshold

            arr[j] = np.expand_dims(data, axis=-1)
            return 1

        return _cln

    return _climate_normal, shape, None


def sample_date(dataset):
    """
    An auxiliary function for use with make_generator
    creates an auxiliary output consisting of the date of each sample in the batch
    Dates (datatype) are specified as datetime.date objects
    :param dataset: DatasetManager object.
                    Source of raw data.
    :return: tuple(function pointer, shape, datatype)
    """
    shape = [1]

    def _sample_date(arr):
        """
        creates a function for internal use with worktions in _generator
        :param arr: placeholder np.ndarray
        :return:  function pointer
        """

        def _sd(arg):
            i, j = arg
            arr[j] = dataset.all_dates[i]
            return 1

        return _sd

    return _sample_date, shape, date
