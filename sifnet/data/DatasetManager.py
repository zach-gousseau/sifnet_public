"""
class DatasetManager, very useful for easy management of generic datasets.
Matthew King
September 13th, 2019
@ NRC - OCRE
"""


import os
import numpy as np
from math import ceil
import yaml
import csv
from pprint import pprint as prettyprint
import copy
from datetime import timedelta, date
from concurrent.futures import ThreadPoolExecutor
from pkg_resources import resource_filename
from tqdm import tqdm
from sifnet.data.data_manager.create_datasets import create_era_interim_dataset, create_era5_dataset, create_ostia_dataset
from sifnet.data.data_manager.dataset_functions import normalize_feature, calculate_degree_days, create_landmask
from sifnet.data.GeneratorFunctions import historical_all_channels, future_single_channel_thresholded, climate_normal


class DatasetManager:
    """
    DatasetManager: A class encompassing all the messy details of loading and preparing the dataset.
    May be used to work with existing datasets or to generate a new dataset given a valid YAML config file.

    When initializing the only argument may take one of two forms:
                       1) A dictionary containing vital configuration info.
                       2) Tuple of (config dictionary, path to YAML config file)
    Option 2 is preferable when generating a new dataset, to enable saving metadata back to the YAML file.

    When a new dataset is being generated, each individual variable is saved to its own file in the directory:
        /work/local_data/vars/{variable unique descriptor filename}.npy
    Additionally, before generating a variable the manager will check this directory for any equivalent variables
    which may have already been generated. If such a variable exists, it will be loaded rather than re-processing
    the same data.
    """

    def __init__(self, config, source='./raw_datasets/', pre_computed_path="./precomputed"):
        """
        Instantiate the DatasetManager (DSM) from a given configuration. This configuration will tell the
        DSM where to find the numpy datacube AND how to generate the datacube if it does not already exist.

        The configuration should be a dictionary. This dictionary should (but does not have to) come from a
        YAML file found somewhere else in the project directory structure.

        The argument may also be passed as a tuple: (config_dictionary, string_file_path).
        This alternative allows the DSM to write back to the source YAML file. This is desirable whenever
        generating a datacube for the first time, so that certain metadata such as variable means, standard
        deviations, and also the raster resolution (if not previously specified).


        :param config: dataset configuration
                        may have the form of a dictionary OR tuple(dictionary, filepath)
                        dictionary must have attributes {'name', 'source_file', 'years', 'yearly_start', 'yearly_end'}
                        filepath should point to original config file used to produce dictionary.

        :param source: String, optional (and not advised). Path to directory from which NetCDF files may be found
                        associated with each source and type.
                        Assumed that subdirectories OSTIA, ERA5, and ERAI exist.

        """

        if type(config) == tuple and len(config) > 1:
            config_filepath = config[1]
            config = config[0]
        elif type(config) == dict:
            config_filepath = []
        else:
            raise ValueError('Invalid input. Must be tuple(config, filepath) or dict(config)')

        print('Creating DatasetManager with the following configuration:')
        prettyprint(config)

        # check if the provided dict has all necessary parameters
        missing_parameters = []
        for param in {'name', 'data_path', 'years', 'yearly_start', 'yearly_end',
                      'landmask_threshold'}:
            if param not in config:
                missing_parameters.append(param)

        if missing_parameters:  # if set is not empty
            raise ValueError("Config missing critical parameters: {}".format(missing_parameters))

        self.data_source_path = source
        self.pre_computed_path = pre_computed_path

        self.config_filepath = config_filepath
        self.name = config['name']
        self.data_path = config['data_path']
        self.landmask_threshold = config['landmask_threshold']

        self.years = config['years']

        self.start_yearly = config['yearly_start']  # e.g [9,1] for September 1st
        self.end_yearly = config['yearly_end']      # e.g [12,31] for December 31st
        self.num_vars = len(config['variables'])    # Number of data channels
        self.lat_bounds = config['lat_bounds']
        self.long_bounds = config['long_bounds']

        # validate formatting of config
        if (len(self.start_yearly) != 2 or len(self.end_yearly) != 2
                or type(self.start_yearly[0]) != int or type(self.end_yearly[0]) != int
                or type(self.start_yearly[1]) != int or type(self.end_yearly[1]) != int):
            raise TypeError("Invalid format of yearly start and/or end date")

        if (type(self.years) != list and type(self.years) != np.ndarray) or type(self.years[0]) != int:
            raise TypeError("Invalid format of years")

        if self.landmask_threshold < 0 or self.landmask_threshold > 1:
            raise ValueError("Landmask threshold must be between 0.0 - 1.0")

        self.all_dates = []

        next_day = timedelta(1)  # used to update 'current' to the next day.
        
        # create list of all dates between start_yearly and end_yearly for each year
        for year in self.years:
            current = date(year, self.start_yearly[0], self.start_yearly[1])  # date(year, month, day)

            # catch wrap-around (e.g. when 'ending' in January of following year)
            if self.end_yearly[0] < self.start_yearly[0]:
                end = date(year+1, self.end_yearly[0], self.end_yearly[1])  # date(year, month, day)
            else:
                end = date(year, self.end_yearly[0], self.end_yearly[1])  # date(year, month, day)

            while end >= current:  # see note below on comparison of dates.
                self.all_dates.append(current)
                current = current + next_day

        self.all_dates = np.array(self.all_dates)
        self.raw_data = []

        # set the raster size using yaml config if possible
        if 'raster_size' not in config:
            self.resolution = None
        else:
            self.resolution = config['raster_size']

        # if processed input data .npy file exists, use it
        if os.path.exists(self.data_path):
            self.raw_data = np.load(self.data_path)

        # otherwise, create the processed .npy file from scratch using raw data
        elif os.path.exists(os.path.dirname(self.data_path)):
            print("The specified dataset: \"{}\" does not exist. Create dataset now?".format(self.name))
            answer = 0

            while answer not in {'Yes', 'No', 'yes', 'no'}:
                answer = 'yes'  # input("Enter Yes or No: ")

            if answer in {'Yes', 'yes'}:
                print("CREATING DATASET")
                self.raw_data = self._create_dataset(config, config_filepath)
            else:
                raise IOError("File {} does not exist. Fix config or generate dataset.".format(self.data_path))
        else:
            raise IOError("Directory {} does not exist. \
                          Fix config or create directory".format(os.path.dirname(self.data_path)))

        print('\nProcessed dataset has resulting shape: {}'.format(self.raw_data.shape))

        # shape checking
        assert(len(self.all_dates) == len(self.raw_data)), 'Length(first dimension) of numpy cube no longer matches ' \
                                                           'expected length based on specified date range and years' \
                                                           '\n Consider reverting the yaml file or deleting the ' \
                                                           'associated .npy file.'

        assert(self.num_vars == self.raw_data.shape[-1]), 'The final dimension of numpy cube no longer matches' \
                                                          'the number of variables specified in yaml. \n' \
                                                          'Consider reverting the yaml config file or deleting the ' \
                                                          'assocated .npy file.'

        # set yaml config as attribute
        self.dataset_config = config

        # check for dates csv file and write one if it does not exist
        # csv file format: date,numpy_join_index
        dates_path = os.path.join(os.path.dirname(config['data_path']), "{}_dates.csv".format(config['name']))
        try:
            if not os.path.exists(dates_path):
                date_strings = [str(d) for d in self.all_dates]
                with open(dates_path, 'w') as file:
                    writer = csv.writer(file)
                    writer.writerow(['date', 'numpy_join_index'])
                    for i in range(len(date_strings)):
                        d = date_strings[i]
                        writer.writerow([d, i])
                del date_strings
        except IOError as e:
            print(e)
            print('Encountered IO error during attempt to write dates csv file. Carrying on as normal.')

        # un-initialized until a call to config()
        # empty placeholder lists
        self.train_dates = []
        self.val_dates = []
        self.test_dates = []

        self.train_lookup_table = []
        self.val_lookup_table = []
        self.test_lookup_table = []

        self.train_years = []
        self.val_years = []
        self.test_years = []

        self.days_of_historic_input = 0
        self.forecast_days_forward = 0

        self.true_start_yearly = []
        self.true_end_yearly = []

        self.bad_dates_lookup_table = np.array([], dtype=int) #self._check_for_invalid_data()
        self.bad_dates = self.all_dates[self.bad_dates_lookup_table]

        self.configured = False

    def find_landmask_channel(self):
        """
        Searches through the dataset's config (yaml) and finds the channel id of the landmask variable.
        Raises ValueError is no such channel exists.
        :return: int, ID/index of the landmask channel.
        """
        config = self.dataset_config
        for var in config['variables']:
            if var['src'] == 'static' and var['type'] == 'landmask':
                return int(var['id'])
        # outside loop, if this point is reached, there was no landmask.
        raise ValueError('Could not find dataset landmask channel')

    def add_bad_dates(self, bad_dates):
        """
        Allows the user to manually specify dates which should not be used.
        Dates are specified using datetime.date objects.
        List of training, val, and test dates will be updated accordingly.
        :param bad_dates: list or array of dates which will be added to bad dates.
        """
        if (type(bad_dates) != list and type(bad_dates) != np.array) or type(bad_dates[0]) != date:
            raise TypeError("Dates should be a list of bad_dates")

        temp = list(self.bad_dates_lookup_table)
        for d in bad_dates:
            ind = self.find_index_of_date(d)  # find index of the in_date
            temp.append(ind)

        temp = sorted(temp)
        self.bad_dates_lookup_table = np.array(temp)
        self.bad_dates = self.all_dates[self.bad_dates_lookup_table]

        self._update_dates()

    def config(self, days_of_historic_input, forecast_days_forward, validation_years,
               test_years, use_default_range=True, custom_date_range=None,
               train_remainder=True, custom_train_years=None):
        """
        Config
        DatsetManager must be configured prior to creating any generators.
        DatasetManager may be configured any number of times, which may be useful if changing validation_years
        as part of a k-fold cross validation strategy.

        :param days_of_historic_input: Number of days of input data (past and 'current' day)
        :param forecast_days_forward: Number of days to forecasts
        :param validation_years: list or array of years for validation e.g. [2000, 2005, 2016]
        :param test_years:  list or array of years for testing e.g [2001, 2006, 2017]
        :param use_default_range: If the full range of yearly available data should be used. Default True.
        :param custom_date_range: tuple of tuples. ((start_date),(end_date)) e.g ((9,1),(12,31))
                                  Must be specified only if use_default_range = False
        :param train_remainder: If all remaining years should be used in train_years. Default True.
        :param custom_train_years: List or array of years. e.g [1999, 2001, 2002]
                                   Must be specified only if train_remainder = False
                                   Must have no overlap with test_year or validation_years.
        """

        self.days_of_historic_input = days_of_historic_input
        self.forecast_days_forward = forecast_days_forward

        # validate argument
        if not use_default_range:
            if not custom_date_range:  # if set is empty
                raise ValueError("custom_date_range must be non-empty if full_range=False")
            elif type(custom_date_range) != tuple or type(custom_date_range[0]) != tuple or \
                    type(custom_date_range[0][0]) != int:
                raise TypeError("data range incorrectly typed. Must be tuple of tuple of ints")

            # used to check validity of provided custom start/end dates.
            if len(custom_date_range) == 2:
                custom_start = custom_date_range[0]
                custom_end = custom_date_range[1]
            else:
                raise ValueError("custom_date_range must have exactly two elements. "
                                 "Received {}".format(len(custom_date_range)))

            if self.end_yearly[0] < self.start_yearly[0]:
                default_end = date(self.years[0] + 1, self.end_yearly[0], self.end_yearly[1])  # in_date(year, month, day)
            else:
                default_end = date(self.years[0], self.end_yearly[0], self.end_yearly[1])  # in_date(year, month, day)

            # check for wrap-around such as ending data-range in January
            if custom_end[0] < custom_start[0]:
                custom_end_test = date(self.years[0]+1, custom_end[0], custom_end[1])
            else:
                custom_end_test = date(self.years[0], custom_end[0], custom_end[1])

            custom_start_test = date(self.years[0], custom_end[0], custom_end[1])
            default_start = date(self.years[0], self.start_yearly[0], self.start_yearly[1])  # e.g in_date(1985, 9, 1)

            # A note regarding comparison of date.
            # The magnitude of a date is greater than that of all previous calendar date
            # e.g date(2017,9,1) > date(2016, 12, 31) --> True

            # If custom end is later than default
            if custom_end_test > default_end:
                raise ValueError("Custom data range cannot end after dataset yearly_end")

            # If custom start is earlier than default
            if default_start > custom_start_test:
                raise ValueError("Custom data range cannot begin prior to yearly_start")

            # input valid if made it this far
            local_start = custom_start
            local_end = custom_end
        else:
            local_start = self.start_yearly
            local_end = self.end_yearly

        # validate input 'validation_years'
        if len(validation_years) > 0:
            if type(validation_years) != list or type(validation_years[0]) != int:
                raise TypeError("validation_years must be a list of integers")
            else:
                for year in validation_years:
                    if year not in self.years:
                        raise ValueError("Validation year {} is not valid".format(year))

        # validate input 'test_years'
        if len(test_years) > 0:
            if type(test_years) != list or type(test_years[0]) != int:
                raise TypeError("test_years must be a list of integers")
            else:
                for year in test_years:
                    if year not in self.years:
                        raise ValueError("Test year {} is not valid".format(year))
                    if year in validation_years:
                        print("Test year {} is also in validation_years".format(year))

        # validate training year argument(s)
        if not train_remainder:
            if not custom_train_years:
                raise ValueError("custom_train_years must be non-empty if train_remainder=False")
            elif (type(custom_train_years) != list and type(custom_train_years) != np.ndarray) \
                    or type(custom_train_years[0]) != int:
                raise TypeError("custom_train_years incorrectly formatted")
            else:
                for year in custom_train_years:
                    if year not in self.years or year in test_years or year in validation_years:
                        raise ValueError("Custom train year {} is not valid".format(year))
            # Valid if made it here
            training_years = custom_train_years
        else:
            training_years = [year for year in self.years if year not in validation_years and year not in test_years]

        # each of the tables are first populated by _calc_index_from_daterange.
        # These initial tables are over-optimistic. The call to _update_dates() then corrects by checking
        # between these tables and any detected bad-dates.

        self.train_lookup_table = self._calc_index_from_daterange(training_years, local_start, local_end)
        self.val_lookup_table = self._calc_index_from_daterange(validation_years, local_start, local_end)
        self.test_lookup_table = self._calc_index_from_daterange(test_years, local_start, local_end)

        self.train_years = np.array(training_years, dtype=np.int)
        self.val_years = np.array(validation_years, dtype=np.int)
        self.test_years = np.array(test_years, dtype=np.int)

        self.true_start_yearly = local_start
        self.true_end_yearly = local_end

        self._update_dates()

        # implementation of computing climate normals for a kfold training evaluation
        # checks to see if climate_normals are requested, updates zeroed channel with climate_normals
        if self.configured:

            clim_norm = False
            with open(self.config_filepath) as f:
                temp_config = yaml.safe_load(f)

            temp_vars = temp_config['variables']
            for i, temp_var in enumerate(temp_vars):
                if temp_var['type'] == 'siconc':
                    siconc_channel = i
                if temp_var['src'] == 'climate_norm':
                    norm_channel = i
                    clim_norm = True

            if clim_norm:
                print('Computing climate normals...')

                """
                To provide clarification on how this algorithm is designed to work:
                
                We initialize an array with form [year, month, day, x, y] and fill it with 2's. The array is then filled
                in with the corresponding data from self.raw_data. After populating the array, we can see what day's we
                actually have data for; for example, if train_raw[2000, 3, 2, 0, 0] has a number belonging to [0, 1],
                then we have a valid data point; if train_raw[2000, 3, 2, 0, 0] == 2, we treat this value as null. The
                train_raw array was not initialized with the np.empty() method because it actually just fills an array
                with very small values (which do belong to [0, 1], and thus can not be distinguished from valid data
                points).
                
                This strategy handles months with different days of the year. For months with 30 days, the last item 
                along that dimension will be filled with 2's and treated as null.
                
                We then initialize the train_norm array and fill it with 2's (same idea). The train_norm array takes
                form [month, day, x, y] and is populated with averages of train_raw along the year axis. Data points 
                with value 2 are not considered during averages.
                
                Finally, we return to self.raw_data; at each day along the day dimension, the month and day are
                gathered, and we update that raster with the corresponding (same day and month) raster from train_norm.
                """

                # initializing array that holds sea ice concentration rasters for each year, month, day
                train_raw = np.ones(shape=(len(self.train_years) + 10, 12, 31, self.resolution[0], self.resolution[1])) * 2

                year = 0
                curr_year = self.train_dates[0].year
                for index, date in enumerate(self.train_dates):
                    if curr_year == self.train_dates[index].year:

                        # filling in train_raw with values
                        train_raw[year, date.month - 1, date.day - 1, :, :] = self.raw_data[self.train_lookup_table[index], :, :, siconc_channel]

                        # applying ice concentration threshold of 0.15
                        train_raw[year, date.month - 1, date.day - 1, :, :] = (train_raw[year, date.month - 1, date.day - 1, :, :] > 0.15).astype('uint8')
                    else:

                        # moving to next year
                        curr_year = self.train_dates[index].year
                        year += 1
                        train_raw[year, date.month - 1, date.day - 1, :, :] = self.raw_data[self.train_lookup_table[index], :, :, siconc_channel]

                        # applying ice concentration threshold of 0.15
                        train_raw[year, date.month - 1, date.day - 1, :, :] = (train_raw[year, date.month - 1, date.day - 1, :, :] > 0.15).astype('uint8')

                # initializing array that holds normals for each month, day of the year
                train_norm = np.ones(shape=(12, 31, self.resolution[0], self.resolution[1])) * 2

                for x in range(self.resolution[0]):
                    for y in range(self.resolution[1]):
                        for day in range(31):
                            for month in range(12):

                                # gathering all valid data points (not 2's) for a given x, y, day, month
                                temp_train = [train_raw[year, month, day, x, y] for year in range(len(self.train_years)) if train_raw[year, month, day, x, y] != 2]
                                if len(temp_train) != 0:

                                    # computing mean and updating train_norm
                                    train_norm[month, day, x, y] = np.mean(temp_train)
                                    # applying 0.5 threshold and converting from bool to int
                                    train_norm[month, day, x, y] = (train_norm[month, day, x, y] > 0.5).astype('uint8')

                # updating self.raw_data with corresponding raster of normals
                for i in range(self.raw_data.shape[0]):
                    self.raw_data[i, :, :, norm_channel] = train_norm[self.all_dates[i].month - 1, self.all_dates[i].day - 1, :, :]

                print('Done computing climate normals')

        self.configured = True

    def _update_dates(self):
        """
        Internal function. Updated the list of dates for training, validation, and testing.
        Starts with the default per-set lookup tables and uses _validate_and_update to remove any unusable data.

        Then uses the updated lookup tables to generate the list of associated dates.
        """

        self.train_lookup_table = self._validate_and_update(self.train_lookup_table)
        self.val_lookup_table = self._validate_and_update(self.val_lookup_table)
        self.test_lookup_table = self._validate_and_update(self.test_lookup_table)

        self.train_dates = self.all_dates[self.train_lookup_table]
        self.val_dates = self.all_dates[self.val_lookup_table]
        self.test_dates = self.all_dates[self.test_lookup_table]

    def find_index_of_date(self, this_date):
        """
        Binary search to find specified date. Non-recursive.
        :param this_date: a datetime.date, must exist in self.all_dates.
        :return: index such that self.all_dates = this_date
        """
        assert(type(this_date) == date)
        left = 0
        right = len(self.all_dates) - 1
        while left <= right:
            midpoint = int(left + (right-left)/2)
            if self.all_dates[midpoint] < this_date:
                left = midpoint + 1
            elif self.all_dates[midpoint] > this_date:
                right = midpoint - 1
            else:
                return midpoint
        raise Exception("Could not find provided in_date {} in self.all_dates".format(this_date))

    def _calc_index_from_daterange(self, years, begin, end):
        """
        Internal function which calculates the index of dates across specified years
        falling between yearly begin and yearly end.
        :param years: list or array of years e.g. [2000, 2001, ...]
        :param begin: tuple of ints, the yearly beginning point e.g (9,1) (month,day)
        :param end: tuple of ints, the yearly end point e.g. (12,31) (month,day)
        :return: np.array of indices
        """
        arr = []
        wrap_around = bool(end[0] < begin[0] or (end[0] == begin[0] and end[1] < begin[1]))

        for year in years:
            start = date(year, begin[0], begin[1])  # (year, month, day)
            if wrap_around:
                final = date(year+1, end[0], end[1])
            else:
                final = date(year, end[0], end[1])

            start = self.find_index_of_date(start)
            final = self.find_index_of_date(final)
            for i in range(start, final+1):
                arr.append(i)

        return np.array(arr)

    def _validate_and_update(self, samples):
        """
        Internal method. Iterates through a set of sample dates (indicies)
            and checks for any samples which are not usable.

        Unusable data includes any data which overlaps with 'bad' data. (currently not enabled 12/16/2019)
        Unusable data includes any date where date + forecast_days_forward > yearly endpoint.

        :param samples: an array or list of the indexes of potential samples
        :return: an updated list of indexes of the same or less length as samples
        """
        if len(self.bad_dates_lookup_table) > 0:
            # remove any samples which have overlap in their data-range with any 'bad' data.
            samples = [x for x in samples
                       if not np.isin(self.bad_dates_lookup_table,
                                      np.arange(x - self.days_of_historic_input + 1,
                                                x + self.forecast_days_forward + 1)).any()]

        # ensure samples (+forecast range) do not exceed end of dataset
        samples = [x for x in samples if self._is_valid_sample_date(self.all_dates[x])]

        samples = np.array(samples, dtype=np.int)

        return samples

    def _check_for_invalid_data(self):
        """
        Internal method. Checks ice concentration data for multiple consecutive days of identical data.
        Current threshold is more than two days in a row of identical data.
        :return: list of invalid dates (by index)
        """
        invalid_dates = []
        in_a_row = 0
        for i in range(1, len(self.all_dates)-1):
            if np.array_equal(self.raw_data[i, :, :, 0], self.raw_data[i - 1, :, :, 0])\
                    and np.array_equal(self.raw_data[i, :, :, 0], self.raw_data[i + 1, :, :, 0]):

                in_a_row = in_a_row + 1

                if in_a_row > 2:  # threshold up for debate.
                    invalid_dates.append(i)
            else:
                in_a_row = 0

        return np.array(invalid_dates, dtype=np.int)

    def _one_repeat_cleanup(self):
        """
        Internal function. Replicated the functionality of "ostia_correct_errorneous_ice_samples"
        Also with consideration of 'bad' dates which are completely unrecoverable.
        """
        for i in range(1, len(self.all_dates) - 1):
            if(np.array_equal(self.raw_data[i, :, :, 0], self.raw_data[i - 1, :, :, 0])
                    and i not in self.bad_dates_lookup_table):
                self.raw_data[i, :, :, 0] = np.mean([self.raw_data[i - 1], self.raw_data[i + 1]], axis=0)

    @staticmethod
    def is_leap_year(year):
        """
        :param year: integer year
        :return: boolean, true if year is a leap year.
        """

        return bool(not (year % 4) and ((year % 100) or not (year % 400)))

    def _is_valid_sample_date(self, in_date):
        """
        Internal function. A in_date is 'valid' if the dataset includes data in the range centered around that in_date.
        :param in_date: datetime.in_date object
        :return: boolean, true if in_date is a valid sample
        """
        year = in_date.year
        begin = self.true_start_yearly
        end = self.true_end_yearly
        wrap_around = bool(end[0] < begin[0] or (end[0] == begin[0] and end[1] < begin[1]))

        if wrap_around and in_date.month >= begin[0]:
            end = date(year + 1, end[0], end[1])
        else:
            end = date(year, end[0], end[1])

        if wrap_around and in_date.month < begin[0]:
            begin = date(year - 1, begin[0], begin[1])
        else:
            begin = date(year, begin[0], begin[1])

        delta_forward = timedelta(self.forecast_days_forward + 1)
        delta_backward = timedelta(self.days_of_historic_input - 1)

        return bool(((in_date - delta_backward) >= begin) and ((in_date + delta_forward) <= end))

    def query_data_from_daterange(self, start, end, years=None):
        """
        Returns the value of the datacube for all data falling in the specified date range.
        Not a part of the primary pipeline.
        :param start: tuple or list of length 2.
               Specifies the begin of yearly range (inclusive)
        :param end: tuple or list of length 2.
               Specifies the end of yearly range (inclusive)
        :param years: optional, list of years
               Subset of years for which data will be queried. Default full range of available years.
        :return: tuple(data, dates)
        """
        if type(start) != tuple and type(start) != list:
            raise TypeError('Start must be a tuple or list. Received type {}'.format(type(start)))

        if type(end) != tuple and type(end) != list:
            raise TypeError('End must be a tuple or list. Received type {}'.format(type(end)))

        if len(start) != 2:
            raise ValueError('Received start of length not equal to 2.')

        if len(end) != 2:
            raise ValueError('Received end of length not equal to 2.')

        if type(start[0]) != int or type(start[1]) != int:
            raise TypeError('Received start with values of non-int type. Received of type ({},{})'.format(
                type(start[0]), type(start[1])))

        if type(end[0]) != int or type(end[1]) != int:
            raise TypeError('Received end with values of non-int type. Received of type ({},{})'.format(
                type(end[0]), type(end[1])))

        if start[0] < 1 or start[0] > 12 or start[1] < 1:
            raise ValueError('Received start with month less than 1 or greater than 12, or with day less than 1')

        if end[0] < 1 or end[0] > 12 or end[1] < 1:
            raise ValueError('Received end with month less than 1 or greater than 12, or with day less than 1')

        if years is not None:
            if type(years) != list:
                raise TypeError('Received years of non-list type')

            for year in years:
                if type(year) != int:
                    raise TypeError('Received year {} of non-int type'.format(year))

                if year not in self.years:
                    raise ValueError('Year {} is not in the dataset'.format(year))
        else:
            years = self.years

        # end input validation
        query_lookup_table = []
        wrap_around = bool(end[0] < start[0] or (end[0] == start[0] and end[1] < start[1]))
        dates_ = self.all_dates
        print('Making query')
        for index in tqdm(range(len(self.raw_data))):
            this_date = dates_[index]
            y = this_date.year
            m = this_date.month
            d = this_date.day
            if y in years or (wrap_around and (y-1) in years):
                if (m >= start[0] and m <= end[0]) or (wrap_around and (m >= start[0] or m <= end[0])):
                    if m != start[0] and m != end[0]:
                        # add because this month is neither of the endpoint months, so we get all days in the month
                        query_lookup_table.append(index)
                    elif start[0] != end[0] and m == start[0] and d >= start[1]:
                        # add because start and end are not same month, it is start month,
                        # and day is greater than start day
                        query_lookup_table.append(index)
                    elif start[0] != end[0] and m == end[0] and d <= end[1]:
                        # add because start and end are not same month, it is end month,
                        # and day is less than end day
                        query_lookup_table.append(index)
                    elif start[0] == end[0] and m == start[0] and start[1] <= d <= end[1]:
                        # add because start and end are same month, it is that month,
                        # and day is between start and end day
                        query_lookup_table.append(index)

        query_lookup_table = np.array(query_lookup_table)
        data = self.raw_data[query_lookup_table]
        dates_ = self.all_dates[query_lookup_table]
        assert len(data) == len(dates_)
        return data, dates_

    def make_generator(self, batch_size=16, option='train', input_functions=None,
                       output_functions=None, auxiliary_functions=None,
                       multithreading=False, safe=True):
        """
        Creates a generator object which will yeild specific samples based on the current configuration and
        available data. Batch size represents the number of samples that are yielded at a time.
        Option specifies which subset of the data from which samples should be obtained.
        The input, output, and auxiliary functions define how the data should be processes.
        See GeneratorFunctions.py
        Multithreading allows the individual functions to be completed in parallel.
        Safe ensures that the yielded data is not modified after being yielded.

        Depending on the current task (what is being trained) data may need to be processed in very different ways.
        This highly dynamic design with specifiable input and output (and aux) functions allow for the generator
        to be built so as to compute any arbitrary pre-processing without having to re-write the pipeline.

        :param batch_size: desired batch size
        :param option: The desired generator. Either 'train', 'val', or 'test'
        :param input_functions: optional list of custom input functions
        :param output_functions: optional list of tuples (function, sample_shape)
        :param auxiliary_functions: optional list of tuples (function, sample_shape, datatype)
        :param multithreading: optional boolean. Default False. Controls if multithreading is to be used.
        :param safe: optional boolean. Default True. Controls if deepcopy is used when yielding results.
                        Recommended to ensure that results are not corrupted as the next batch is prepared.
        :return: tuple(generator object, batches per epoch)
        """

        if not self.configured:
            raise Exception('Cannot create generators prior to configuration.')

        if option not in {'train', 'val', 'test'}:
            raise ValueError("option must be one of 'train', 'val', or 'test'.")

        if option == 'train':
            table = self.train_lookup_table
            shuffle = True
        elif option == 'val':
            table = self.val_lookup_table
            shuffle = False
        else:
            table = self.test_lookup_table
            shuffle = False

        # TODO: try np.float32
        # datatype of samples being fed into keras fit_generator
        default_datatype = np.float64

        # we store the shapes and data types of the arrays which must be allocated for each generator function.
        input_shapes = []
        input_datatypes = []
        if not input_functions:
            # load defaults if none specified
            fun, shape, dt = historical_all_channels(self)  # default
            input_functions = [fun]
            input_shapes.append(shape)
            if dt is not None:
                input_datatypes = [dt]
            else:
                input_datatypes = [default_datatype]
        else:
            temp = input_functions
            input_functions = []
            for X in temp:
                if not(len(X) == 3):
                    raise ValueError("Length of each input_function tuple must be 3.")
                fun, shape, dt = X
                input_functions.append(fun)
                input_shapes.append(shape)
                if dt is not None:
                    input_datatypes.append(dt)
                else:
                    input_datatypes.append(default_datatype)

        output_shapes = []
        output_datatypes = []
        if not output_functions:
            fun, shape, dt = future_single_channel_thresholded(self)  # default
            output_functions = [fun]
            output_shapes = [shape]
            if dt is not None:
                output_datatypes = [dt]
            else:
                output_datatypes = [default_datatype]
        else:
            temp = output_functions
            output_functions = []
            for X in temp:
                if not(len(X) == 3):
                    raise ValueError("Length of each output_function tuple must be 3")
                fun, shape, dt = X
                output_functions.append(fun)
                output_shapes.append(shape)
                if dt is not None:
                    output_datatypes.append(dt)
                else:
                    output_datatypes.append(default_datatype)

        # auxiliary functions are for use in debugging or other but not during training or evaluation.
        # There are no defaults because by default there are no auxiliary functions.
        # If provided, generator yields list of length 3 instead of length 2.
        aux_shapes = []
        aux_datatypes = []
        if auxiliary_functions:
            temp = auxiliary_functions
            auxiliary_functions = []
            for X in temp:
                if not(len(X) == 3):
                    raise ValueError("Length of each auxiliary_function tuple must be 3")
                else:
                    fun, shape, dt = X
                    auxiliary_functions.append(fun)
                    aux_shapes.append(shape)
                    if dt is not None:
                        aux_datatypes.append(dt)
                    else:
                        aux_datatypes.append(default_datatype)

        # it may be necessary to pad the number of samples so that it is evenly divided by the batch size.
        # this has the consequence that a few samples may be yielded twice per epoch.
        num_batches = ceil(len(table) / batch_size)
        if len(table) % batch_size:  # if batch_size does not evenly divide number of samples
            last_batch_padding = batch_size - (len(table) % batch_size)  # amount of padding (non-negative integer)
        else:
            last_batch_padding = 0

        # create the actual generator
        return self._generator(table, batch_size, num_batches, last_batch_padding,
                               shuffle, input_functions, output_functions, auxiliary_functions,
                               input_shapes, output_shapes, aux_shapes, input_datatypes, output_datatypes,
                               aux_datatypes, multithreading, safe), num_batches

    def _generator(self, dates, batch_size, num_batches, last_batch_pad, shuffle,
                   input_functions, output_functions, auxiliary_functions, input_shapes, output_shapes,
                   auxiliary_shapes, input_datatypes, output_datatypes, auxiliary_datatypes,
                   multithreading=True, safe=True):
        """ Internal generator function.

        :param dates: array/list of sample dates (actually used for array indexing)
        :param batch_size: int, desired batch size
        :param num_batches: int, number of batches per epoch
        :param last_batch_pad: int, amount padding to add so that num_batches evenly divides len(bad_dates)
        :param shuffle: boolean.
        :param input_datatypes: list of data types
                e.g. [np.float64, np.float32, np.uint8]
        :param output_datatypes: list of data types
        :param auxiliary_datatypes: list of data types
        :return: one batch per call of next()
                 A tuple of either length 2 or length 3, depending if any auxiliary functions were provided
        """
        np.random.seed(123987)  # arbitrary but improves reproducibility

        local_array = np.zeros(num_batches*batch_size, dtype=np.int)  # holds indicies

        work_functions = []  # list of functions that must be executed for each batch

        # create list of placeholder arrays
        input_arrays = []
        target_arrays = []

        for i in range(len(input_functions)):
            # each function has an associated shape and datatype.
            # The shape must be modified so as to accommodate the batch-size.
            func = input_functions[i]
            shape = input_shapes[i]
            dt = input_datatypes[i]
            array = np.ndarray(shape=([batch_size] + list(shape)), dtype=dt)
            func = func(array)  # returns a function-pointer
            input_arrays.append(array)
            work_functions.append(func)  # add to work functions

        for i in range(len(output_functions)):
            # each function has an associated shape and datatype.
            # The shape must be modified so as to accommodate the batch-size.
            func = output_functions[i]
            shape = output_shapes[i]
            dt = output_datatypes[i]
            array = np.ndarray(shape=([batch_size] + shape), dtype=dt)
            func = func(array)
            target_arrays.append(array)
            work_functions.append(func)

        if auxiliary_functions:
            auxiliary_arrays = []
            for i in range(len(auxiliary_functions)):
                # each function has an associated shape and datatype.
                # The shape must be modified so as to accommodate the batch-size.
                func = auxiliary_functions[i]
                shape = auxiliary_shapes[i]
                dt = auxiliary_datatypes[i]
                array = np.ndarray(shape=([batch_size]+list(shape)), dtype=dt)
                func = func(array)
                auxiliary_arrays.append(array)
                work_functions.append(func)

            results = (input_arrays, target_arrays, auxiliary_arrays)
        else:
            results = (input_arrays, target_arrays)

        def work_fn(arg):  # function passed into threadPoolExecutor or simple map
            for function in work_functions:
                function(arg)
            return 1

        placeholder = np.arange(batch_size)  # gives each thread it's location in the batch
        pool = ThreadPoolExecutor(batch_size)  # threads which execute work function

        while True:
            if shuffle:
                np.random.shuffle(dates)

            # load most of the array with contents of list
            local_array[:len(dates)] = dates

            # pad the end of the array with a copy of the beginning of the array
            if last_batch_pad > 0:
                local_array[-last_batch_pad:] = local_array[:last_batch_pad]

            # split the array into equal-sized batches. (Guaranteed same size due to padding).
            batches = np.split(local_array, num_batches)
            for batch in batches:

                if multithreading:
                    # each thread processes exactly one element of the batch
                    # each element of the batch is 100% independent of all other elements
                    # thus, there are no race conditions.
                    res = pool.map(work_fn, zip(batch, placeholder))
                else:
                    res = map(work_fn, zip(batch, placeholder))

                for _ in range(batch_size):
                    # res is an iterable object. Calling next() will cause the program to wait
                    # wait until the result is ready, meaning until the work function is complete.
                    next(res)   # ensures the jobs are completed prior to yield.

                if safe:
                    yield copy.deepcopy(results)
                else:
                    yield results

    def _create_dataset(self, config, config_filepath):
        """
        Creates a dataset based on the given config information.
        Prompts user if created dataset should be saved.
        If saved, also updates YAML config to include mean and STD information.
            If running in docker, right-click YAML-file in project explore and
            Deployment-> (Download from <Remote Interpreter Name>) to obtain the updated YAML.
        :param config: configuration dictionary, same as class in constructor
        :param config_filepath: path fo config file
        :return numpy.ndarray data-cube
        """
        missing_parameters = []
        for param in {'lat_bounds', 'long_bounds', 'variables'}:
            if param not in config:
                missing_parameters.append(param)

        if missing_parameters:  # if set is not empty
            raise ValueError("Config missing critical parameters: {}".format(missing_parameters))

        vars = config['variables']
        unsupported_src = []

        for var in vars:
            if var['src'] not in {'ostia', 'erai', 'era5', 'static', 'degree_days', 'climate_norm'}:
                unsupported_src.append(var['id'])

        if unsupported_src:
            raise ValueError("Variable(s) # {} from unsupported source".format(unsupported_src))
        roi = [config['lat_bounds'][0], config['lat_bounds'][1], config['long_bounds'][0], config['long_bounds'][1]]

        arrays = []
        ostia_already_processed = False  # don't want to run separately for sst and ci
        ostia_temp = []  # hold ice and sst if either is needed.

        ostia_directory = os.path.join(self.data_source_path, "OSTIA")
        erai_directory = os.path.join(self.data_source_path,"ERA-INTERIM")
        era5_directory = os.path.join(self.data_source_path,"ERA5")

        # partial_files_directory and solo_var_template are used to save each variable individually.
        # This may be very useful if creation crashes at a later stage, because then each previously generated
        # variable can be quickly loaded rather than fully re-generated.
        partial_files_directory = self.pre_computed_path
        solo_var_template = "VAR_roi-{}_resolution-{}_begin-{}_end-{}_year0-{}_Nyears-{}_src-{}_type-{}.npy"

        resolution = self.resolution

        for i in tqdm(range(len(vars))):
            var = vars[i]
            print(var)
            source = var['src']
            typ = var['type']

            # format the string and remove blank spaces

            solo_var_path = solo_var_template.format(roi, resolution, self.start_yearly, self.end_yearly,
                                                     self.years[0], len(self.years), source, typ).replace(" ", "")
            solo_var_path = os.path.join(partial_files_directory, solo_var_path)

            # check if a variable exists that matches this one.
            # also, resolution must be defined and this must not be a landmask.
            if os.path.exists(solo_var_path) and resolution and source != 'static':
                arr = np.load(solo_var_path)
                print('Loading from file {}'.format(solo_var_path))
            else:
                if source == 'ostia':
                    # Ensure start year is 1985 or greater (no data before 1985)
                    if self.years[0] < 1985:
                        raise ValueError(
                            'OSTIA data starts from 1985. You supplied start year of ',
                            self.years[0])

                    if not ostia_already_processed:
                        ostia_temp = create_ostia_dataset(self.all_dates, ostia_directory, roi, resolution)
                        ostia_already_processed = True
                        print('OSTIA - array shape is: ', ostia_temp[0].shape)

                    if typ == 'ci':
                        arr = ostia_temp[0]
                    elif typ == 'sst':
                        arr = ostia_temp[1]
                    else:
                        raise ValueError("ostia does not support type {}".format(typ))

                elif source == 'erai':
                    # error checking takes place inside
                    arr = create_era_interim_dataset(self.all_dates, erai_directory, roi,  typ, resolution)

                elif source == 'era5':
                    # error checking takes place inside
                    path = "medium_term_ice_forecasting/support_files/era5-landsea-mask.nc"
                    landmask_path = resource_filename("sifnet", path)
                    landmask = create_landmask(landmask_path, self.all_dates,
                                          roi, resolution, self.landmask_threshold)
                    arr = create_era5_dataset(self.all_dates, era5_directory, landmask[0,:,:], roi, typ, resolution)
                    
                elif source == 'degree_days':
                    unprocessed_input = arrays[var['from']]
                    arr = calculate_degree_days(unprocessed_input, self.all_dates, self.start_yearly[0],
                                                self.start_yearly[1], resolution, typ, False)

                elif source == "climate_norm":
                    arr = np.zeros(shape = (len(self.all_dates), resolution[0], resolution[1]))

                # Get resource path from within module
                elif source == 'static' and typ == 'landmask':
                    path = var['path']
                    landmask_path = resource_filename("sifnet", path)
                    if not os.path.exists(landmask_path):
                        raise IOError("Landmask file path does not exist. Please check YAML. You asked for: {} "
                                      .format(landmask_path))

                    if landmask_path.endswith('.npy'):
                        file = np.load(landmask_path)
                        if not resolution:
                            resolution = list(file.shape)
                        arr = np.resize(file, (len(self.all_dates), resolution[0], resolution[1]))

                    elif landmask_path.endswith('.nc'):
                        arr = create_landmask(landmask_path, self.all_dates,
                                              roi, resolution, self.landmask_threshold)

                    print("landmask shape", arr.shape)
                    landmask = arr
                    landmask_id = i

                if not resolution:
                    resolution = [arr.shape[1], arr.shape[2]]
                    self.resolution = resolution
                    # resolution was [], so we need to recreate the filename before saving var.
                    solo_var_path = solo_var_template.format(roi, resolution, self.start_yearly,
                                                             self.end_yearly, self.years[0], len(self.years),
                                                             source, typ).replace(" ", "")
                    solo_var_path = os.path.join(partial_files_directory, solo_var_path)

                if source != 'static':
                    print("Saving var to {}".format(solo_var_path))
                    np.save(solo_var_path, arr)

            arrays.append(arr)
            
        # outside loop. fix landmask for modified siconc cells
        #mask = np.max(arrays[0], axis=0) * arrays[6][0, :, :]
        #mask[mask>0] = 1
        #mask = np.resize(mask, landmask.shape)
        #landmask *= (1-np.uint8(mask))
        #arrays[landmask_id] = landmask

        # outside loop. Normalize features.
        for i in range(len(vars)):
            arr = arrays[i]
            var = vars[i]
            if var['normalize']:
                print('Normalizing var {}'.format(var['type']))
                arr, mean, std_dev = normalize_feature(arr)
                var['mean'] = float(mean)
                var['std_dev'] = float(std_dev)
                arrays[i] = arr

        merged_array = np.stack(arrays, axis=3).astype(np.float32)

        print('Saving data...')
        np.save(config['data_path'], merged_array)

        if 'raster_size' not in config:
            config['raster_size'] = resolution

        if config_filepath:
            with open(config_filepath, 'w') as f:
                yaml.safe_dump(config, f)  # write updated config to yaml
                f.close()
        else:
            print('Could not save back to YAML')

        return merged_array


if __name__ == '__main__':
    """
    This section is meant to act as a demonstration/testing for use of the DatasetManager
    """
    from sifnet.data.GeneratorFunctions import *
    from matplotlib import pyplot as plt
    import time

    with open(resource_filename('sifnet',
                                'medium_term_ice_forecasting/datasets/Hudson/Hudson_Freeze_v2_reduced_extended.yaml')) as f:
        config = yaml.safe_load(f)
    hdf = DatasetManager(config, pre_computed_path="/home/nazanin/work/nas/SIF/vars")
    print("Unusable Data: {}".format(len(hdf.bad_dates)))
    # hdf.add_bad_dates(dates=[date(1985, 11, 3), dates(2011, 9, 21)])  # just for demo
    # print(len(hdf.bad_dates))

    hdf.config(days_of_historic_input=3, forecast_days_forward=30,
               validation_years=[1989, 2000, 2002, 2014, 2016], test_years=[1990, 2003, 2012, 2013, 2017])
    print("Thirty day forward stats:")
    print("Training samples: {}".format(len(hdf.train_dates)))  # 2170
    print("Validation samples: {}".format(len(hdf.val_dates)))  # 505
    print("Test samples: {}".format(len(hdf.test_dates)))   # 451

    ice = hdf.raw_data[172,:,:,0]
    landmask = hdf.raw_data[172,:,:,-1]
    needs_fix = (((ice == 0) - landmask) == 1)
    plt.imshow(needs_fix)
    plt.colorbar()
    plt.show()
    plt.savefig('/home/nazanin/1.png')
