import os
import yaml

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from datetime import date, timedelta
from scipy.stats.stats import pearsonr
from adjustText import adjust_text
from numpy.polynomial.polynomial import polyfit
import copy
import matplotlib.colors as colors

from sifnet.medium_term_ice_forecasting.utilities.postprocessing_tools import extract_test_records, plot_timeseries_for_points
from sifnet.medium_term_ice_forecasting.utilities.visualization import calc_lat_lon_tics_labels
from sifnet.medium_term_ice_forecasting.utilities.ice_trends import load_first_date
from pkg_resources import resource_filename


def plot_heatmap(values, savepath, title, mask_shape, lat_bounds, long_bounds, vmin=0, vmax=1):
    """
    Plot heatmap using 2D grid of values

    :param values: 2D float array
           Grid of values to plot
    :param savepath: string
            Path to save the plot
    """
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Helvetica']
    plt.rcParams['font.size'] = 12
    fig = plt.figure()
#     plt.title(title)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['blue', 'black', 'red'], 128)
    
    #The following 2 lines are for TMLS figures
#     cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['indigo','green','yellow'], 256)
#     cmap = copy.copy(mpl.cm.get_cmap("ocean"))
    cmap.set_bad('xkcd:white')
    cmap.set_under('gainsboro')
    cmap.set_over('gainsboro')

    # color bar scale is set from -1 to 1 (correlation value)
#     plt.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)
#     plt.imshow(values, norm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax), cmap=cmap)
    plt.imshow(values, vmin=vmin, vmax=vmax, cmap=cmap)
#     plt.clim(-10,30)
    cbar = plt.colorbar()
    cbar.set_label("Accuracy")
    (y_tics, y_tic_labels), (x_tics, x_tic_labels) = calc_lat_lon_tics_labels(mask_shape, lat_bounds, long_bounds)

    y_tics = y_tics[::-1]
    plt.xticks(x_tics, x_tic_labels)
    plt.yticks(y_tics, y_tic_labels)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.tight_layout()
    fig.savefig(os.path.join(savepath, title.replace('\n', ' ') + '.png'))
    fig.savefig(os.path.join(savepath, title.replace('\n', ' ') + '.eps'))
    plt.close(fig)


def plot_correlation_map(savepath, landmask, lead_day, forecast_days_forward, days_of_historic_input, lat, long,
                    model='modeled', region_name='Hudson Bay', season='Freeze-up', load=True):
    """
    Plot the correlation map of true freeze-up and breakup date with predicted freeze-up and breakup date, and saves
    the picture

    :param savepath: str
            path to a directory containing models evaluated at each fold
            e.g /work/IcePresence/NWT_Freeze_v2/H3-F30/model_name
    :param landmask: np.ndarray
            binary 2D array representing land
    :param lead_day: int
            the desired forecast day to observe - 1. i.e. if we wanted 30 day forecasts, we need to input 29
    :param forecast_days_forward: int
            number of days model forecasts
    :param days_of_historic_input: int
            number of days historical days observed by the model
    :param lat: [float, float]
            latitude bounds of the model
    :param long: [float, float]
            longitude bound of the model
    :param model: [float, float]
            type of model used to make predictions
                - modeled
                - normals
                - persistence
    :param region_name: string
            name of geographical location
                - Hudson Bay
                - Baffin Bay
                - NWT
    :param season: string
            type of season
                - Freeze-up
                - Breakup
    :param load: bool
            if false, code will generate npy files even if they have already been generated
    """

    _, _, begin, end, _ = extract_test_records(savepath)

    pred_index, truth_index, _, _ = load_first_date(savepath, model, lead_day, forecast_days_forward,
                                                    days_of_historic_input, season, load)

    num_years, x_resolution, y_resolution = pred_index.shape
    correlation_map = np.ndarray([x_resolution, y_resolution])

    for i in range(x_resolution):
        for j in range(y_resolution):
            if not landmask[i][j]:
                p = pred_index[:, i, j]
                t = truth_index[:, i, j]

                freezeup_breakup_not_found(p, t, begin, end)

                if is_same_element(p) or is_same_element(t) and p[0] == t[0]:
                    correlation_map[i][j] = 1  # np.nan if we want to filter out value
                elif is_same_element(p) or is_same_element(t):
                    correlation_map[i][j] = -2
                else:
                    correlation_map[i][j] = pearsonr(p, t)[0]
                    if correlation_map[i][j] < 0:
                        print(i, j)
                        print("predicted", p)
                        print("truth", t)
            else:
                correlation_map[i][j] = np.nan

    correlation_map = np.ma.masked_where(landmask, correlation_map)
    # correlation_map = np.ma.masked_where(correlation_map == np.nan, correlation_map)

    if model == 'modeled':
        name = 'Model'
    elif model == 'normals':
        name = 'Climate Normal'
    elif model == 'persistence':
        name = 'Persistence'
    else:
        raise ValueError("Incorrect model type specified")

    outpath = os.path.join(savepath, "evaluations", "correlation_map")
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    plot_heatmap(np.flipud(correlation_map), outpath,
                 "{0} Day Correlation Between Observations\nand Predictions at {1} Lead Day by {2} "
                 "in {3}".format(season, lead_day + 1, name, region_name), landmask.shape, lat, long)


def plot_accuracy_within_N_days(savepath, landmask, lead_day, N, forecast_days_forward, days_of_historic_input,
                                lat_bounds, long_bounds, model='modeled', season="Freeze-up", load=True):
    """
    Accuracy within N days

    :param savepath: str
            path to a directory containing models evaluated at each fold
            e.g /work/IcePresence/NWT_Freeze_v2/H3-F30/model_name
    :param landmask: np.ndarray
            binary 2D array representing land
    :param lead_day: int
            the desired lead da to observe - 1. i.e. if we wanted 30 day forecasts, we need to input 29
    :param N: int
            number of days the forecast freeze-up/breakup date can be away from the true freeze-up breakup date
    :param forecast_days_forward: int
            number of days forecasted forwards
    :param days_of_historic_input: int
            number of days historical days observed by the model
    :param lat_bounds: [float, float]
            latitude bounds of the model
    :param long_bounds: [float, float]
            longitude bound of the model
    :param model: [float, float]
            type of model used to make predictions
                - modeled
                - normals
                - persistence
    :param season: string
            type of season
                - Freeze-up
                - Breakup
    :param load: bool
            if false, code will generate npy files even if they have already been generated
    """
    if model == 'modeled':
#         title = "Accuracy of Predicted "+ season+ " Date\nWithin {0} Days Using {1} Lead Day"\
#             .format(N, lead_day)
        title = "Model_"+season+ "_{}".format(lead_day)
    elif model == 'normals':
#         title = "Accuracy of Predicted "+ season+ " Date\nWithin {} Days using Climate Normal".format(N)
        title = "Normal_"+season
    elif model == 'persistence':
        title = "Accuracy of Predicted "+ season+ " Date\nWithin {} Days using persistence".format(N)
    else:
        raise ValueError("Model type {} not supported".format(model))

    _, _, pred_date, truth_date = load_first_date(savepath, model, lead_day,
                                                  forecast_days_forward, days_of_historic_input, season, load)

    num_years, x_resolution, y_resolution = pred_date.shape

    n_days_accuracy_map = np.ndarray([x_resolution, y_resolution])
    for i in range(x_resolution):
        for j in range(y_resolution):
            if not landmask[i][j]:
                p = pred_date[:, i, j]
                t = truth_date[:, i, j]
                time_differences = p - t
                acc = 0
                for delta in time_differences:
                    if abs(delta.days) <= N:
                        acc += 1

                n_days_accuracy_map[i][j] = acc/time_differences.shape[0]
            else:
                n_days_accuracy_map[i][j] = 0

    ## Debug code
    #print(f"Model type: {model}")
    #northstar_pred = pred_date[:, 20, 58]
    #northstar_truth = truth_date[:, 20, 58]
    #for pair in zip(northstar_truth, northstar_pred):
    #    print(pair)
    #print("\n")
    ##

    sample_inacc_locs = []
    sample_acc_locs = []
    for i in range(np.size(n_days_accuracy_map,0)):
        for j in range(np.size(n_days_accuracy_map,1)):
            cell_value = n_days_accuracy_map[i][j]
            if cell_value < 0.1:
                sample_inacc_locs.append((i,j))
            elif cell_value > 0.9:
                sample_acc_locs.append((i,j))
            else:
                continue
    
    #### Retrieve time-series for inaccurate points only for now    
    if len(sample_inacc_locs) >= 3:
        # Plot for three randomly chosen points
        sample_inacc_locs = [sample_inacc_locs[i] for i in 
                             np.random.choice(range(len(sample_inacc_locs)), size=3, replace=True)]
        plot_ts_inaccs = False
        if plot_ts_inaccs:
            coords = [index_to_roi(*loc, savepath) for loc in sample_inacc_locs]
            chosen_years = [2007, 2008, 2009]
            plot_timeseries_for_points(savepath, sample_inacc_locs, coords, chosen_years, 
                                        lead_day, days_of_historic_input, model)
    else:
        print("Insufficient inaccurate points found for analytics. Continuing...")
    ####
    
    if len(sample_acc_locs) >= 3:
        sample_acc_locs = [sample_acc_locs[i] for i in 
                           np.random.choice(range(len(sample_acc_locs)), size=3, replace=True)]
    else:
        print("Insufficient accurate points found. Continuing...")
    
    sample_locs = sample_inacc_locs
    sample_locs.extend(sample_acc_locs)
    
    n_days_accuracy_map = np.ma.masked_where(landmask, n_days_accuracy_map)
    savepath = os.path.join(savepath, "evaluations", "N_day_accuracy_map")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    plot_heatmap(n_days_accuracy_map, savepath, title, landmask.shape, lat_bounds, long_bounds)

    return sample_locs


def get_lat_long(savepath, raster_size=None, lat_bounds=None, long_bounds=None):
    """
    Retrieve latitude and longitude from grid file
    :param config: string
        name of configuration file of experiment (i.e. Hudson/Hudson_Freeze_v2_reduced.yaml)
    :return:
        raster size, longitude bound of region, latitude bound of region, width of one cell (in longitude degrees),
        height of one cell (in latitude degrees)
    """
    if raster_size is None or lat_bounds is None or long_bounds is None:
        test_record_path = os.path.join(savepath, 'test_record.yaml')

        with open(test_record_path, 'r') as f:
            test_record = yaml.safe_load(f)

        raster_size = test_record['raster_size']
        lat_bounds = test_record['lat_bounds']
        long_bounds = test_record['long_bounds']

    complete_lat = abs(lat_bounds[1] - lat_bounds[0])
    unit_lat = complete_lat/raster_size[0]

    complete_long = abs(long_bounds[1] - long_bounds[0])
    unit_long = complete_long/raster_size[1]

    return raster_size, long_bounds, lat_bounds, unit_long, unit_lat


def index_to_roi(row_index, col_index, savepath, raster_size=None, lat_bounds=None, long_bounds=None):
    """
    Convert grid location to latitude and longitude
    :param lat: [float, float]
    :param long: [float, float]
    :param config: string
        name of configuration file of experiment (i.e. Hudson/Hudson_Freeze_v2_reduced.yaml)
    :return:
        row location and column location corresponding to specified lat and long in the raster
    """
    raster_size, long_bounds, lat_bounds, unit_long, unit_lat = get_lat_long(savepath, raster_size,
                                                                                         lat_bounds, long_bounds)
    return lat_bounds[0] - row_index * unit_lat, long_bounds[1] + col_index * unit_long


def roi_to_index(lat, long, savepath, raster_size=None, lat_bounds=None, long_bounds=None):
    """
    Convert latitude and longitude to grid location
    :param lat: [float, float]
    :param long: [float, float]
    :param config: string
        name of configuration file of experiment (i.e. Hudson/Hudson_Freeze_v2_reduced.yaml)
    :return:
        row location and column location corresponding to specified lat and long in the raster
    """
    raster_size, long_bounds, lat_bounds, unit_long, unit_lat = get_lat_long(savepath, raster_size,
                                                                            lat_bounds, long_bounds)
    row_loc = int((lat_bounds[0] - lat)//unit_lat)
    col_loc = int((long - long_bounds[1])//unit_long)

    return row_loc, col_loc


def plot_correlation_at_port(savepath, landmask, main_region, location, lead_day, forecast_days_forward, days_of_historic_input,
                                 region_name, raster_size=None, lat_bounds=None, long_bounds=None, season='Freeze-up',
                                 model='modeled', load=True):
    """
    Plot qqplot of predicted freeze-up breakup date against each other. Year range is currently hardcoded in from
    1996 to 2018

    :param savepath: str
            path to a directory containing models evaluated at each fold
            e.g /work/IcePresence/NWT_Freeze_v2/H3-F30/model_name
    :param landmask: np.ndarray
            binary 2D array representing land
    :param location: [float, float]
            coordinates of the location
    :param lead_day: int
            the desired lead da to observe - 1. i.e. if we wanted 30 day forecasts, we need to input 29
    :param forecast_days_forward: int
            number of days forecasted forwards
    :param days_of_historic_input: int
            number of days historical days observed by the model
    :param region_name: string
            the name of the region (i.e. Hudson Bay, Baffin Bay, NWT)
    :param raster_size: [int, int]
            spatial resolution of the model
    :param lat_bounds: [float, float]
            latitude bounds of the model
    :param long_bounds: [float, float]
            longitude bound of the model
    :param season: string
            type of season
                - Freeze-up
                - Breakup
    :param model: [float, float]
            type of model used to make predictions
                - modeled
                - normals
                - persistence
    :param load: bool
            if false, code will generate npy files even if they have already been generated
    """
    row_loc, col_loc = roi_to_index(location[0], location[1], savepath, raster_size, lat_bounds, long_bounds)
    print(region_name)
    print("row location: ", row_loc)
    print("column location ", col_loc)

    # if landmask, return error
    if landmask[row_loc][col_loc]:
        print("({0}, {1}) is land!".format(location[0], location[1]))
        return

    pred_index, truth_index, pred_dates, truth_dates = load_first_date(savepath, model, lead_day,
                                                                       forecast_days_forward, days_of_historic_input, season, load)

    p = list(pred_index[:, row_loc, col_loc])
    t = list(truth_index[:, row_loc, col_loc])
    p_date = list(pred_dates[:, row_loc, col_loc])
    t_date = list(truth_dates[:, row_loc, col_loc])

#     print("\n")
#     print("First debug block")
#     print(f"Truth-predicted index tuples: {list(zip(t,p))}")
#     print(f"Truth-predicted date tuples: {list(zip(t_date,p_date))}")
#     print("\n")
    #input()
    
    inds = [i for i in range(len(p_date)) if (p_date[i]!=date(1900,1,1) and t_date[i]!=date(1900,1,1))]
    p_date = [p_date[i] for i in inds]
    t_date = [t_date[i] for i in inds]
    t = [t[i] for i in inds]
    p = [p[i] for i in inds]

    # Extract the breakup/freeze-up season begin and end dates from test_record.yaml file
    _, _, begin, end, _ = extract_test_records(savepath)
    freezeup_breakup_not_found(p, t, begin, end)

    # collapse years with the same freeze-up dates together for more organized plotting
    years = list(range(1996, 2018))
    years = [years[i] for i in inds]
    years_dict = {}
    remove_ind = -1
    for i in range(len(p)):
        if years[i] == 2010:  # remove 2010 as outlier from port correlation plots
            remove_ind = i
            continue
        k = (t[i], p[i])
        if k not in years_dict:
            years_dict[k] = [str(years[i])]
        else:
            years_dict[k].append(str(years[i]))
    if remove_ind != -1:
        t.pop(remove_ind)
        p.pop(remove_ind)
        t_date.pop(remove_ind)
        p_date.pop(remove_ind)
    # There may be a lot of years grouped at (0, 0).
    years_dict[(0, 0)] = 'Remaining Years'

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Helvetica']
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(8, 8))

    # Check if p or t are empty
    if not p or not t or not p_date or not t_date:
        print(f"No valid predicted and observed {season} dates found. Qq plot cannot be drawn.")
        return

    # if values in p and t are the same, then there is only one distinct data point. Qqplot cannot be drawn
    if max(p) == min(p) and max(t) == min(t):
        print('All values in predicted and observed {} dates are the same. Qq plot cannot be drawn'.format(season))
        return

    sup = max(max(p), max(t))
    inf = min(min(p), min(t))

    # find the earliest and latest freeze-up date, then divide date range accordingly
    
    start_date = date(1900, 1, 1)
    # find index of smallest element and get the corresponding date
    for i in range(len(p)):
        if p[i] == inf:
            start_date = p_date[i]
            break
        if t[i] == inf:
            start_date = t_date[i]
            break
    end_date = date(1900, 1, 1)
    for i in range(len(p)):
        if p[i] == sup:
            end_date = p_date[i]
            break
        if t[i] == sup:
            end_date = t_date[i]
            break
            
    # if start date is date(1900, 1, 1), something went wrong!
    if start_date == date(1900, 1, 1):
        raise ValueError("Unable to find valid minimum date")
   
            
    #excluding certain ports for paper figures
    if season == 'Breakup':
        if region_name == 'Churchill':
            temp_start = date(1900,5,28)
            temp_end = date(1900,8,1)
        elif region_name == 'Inukjuak':
            temp_start = date(1900,5,12)
            temp_end = date(1900,7,25)
        elif region_name == 'Quataq':
            temp_start = date(1900,5,25)
            temp_end = date(1900,8,1)
        else:
            temp_start = start_date
            temp_end = end_date
    else:
        if region_name == 'Churchill':
            temp_start = date(1900,10,30)
            temp_end = date(1900,12,10)
        elif region_name == 'Inukjuak':
            temp_start = date(1900,11,30)
            temp_end = date(1901,1,15)
        elif region_name == 'Quataq':
            temp_start = date(1900,11,15)
            temp_end = date(1901,1,5)
        else:
            temp_start = start_date
            temp_end = end_date
    
    #### What is the point of this code block?
    start_date = start_date.replace(year=1900)
    end_date = end_date.replace(year=1900)
    if (season=='Freeze-up') and (end_date.month<4):
        end_date = end_date.replace(year=1901)
    ####

    ######## Debug code for plotting params
#     print("\n")
#     print("Second debug block")
#     print(f"p indexes: {p}")
#     print(f"t indexes: {t}")
#     print(f"infimum: {inf}")
#     print(f"supremum: {sup}")
#     print(f"predicted dates: {sorted(p_date)}")
#     print(f"truth dates: {sorted(t_date)}")
#     print(f"temp start: {temp_start}")
#     print(f"temp end: {temp_end}")
#     print(f"start date: {start_date}")
#     print(f"end date: {end_date}")
#     #print(f"Season begin: {begin}")
#     #print(f"Season end: {end}")
#     #input()
#     print("\n")
    #########

    diff_days = (temp_start-start_date).days
    #inf += diff_days
    diff_days = (temp_end-end_date).days
    #sup += diff_days

    ### Sams code for making plots appear less 'squished'
    padding = 2
    inf = inf - padding
    sup = sup + padding
    ###
    skip = 10
    label_locations = list(range(int(inf), int(sup), skip))
    
    # get list of date labels at a certain corresponding to inf and sup
    date_range = [temp_start + timedelta(days=x) for x in range(0, skip*len(label_locations), skip)]
    date_labels = [x.strftime("%b %d") for x in date_range]

    plt.xlim(inf, sup)
    plt.ylim(inf, sup)
    b, m = polyfit(t, p, 1)
#     plt.plot(np.array(t), b + m * np.array(t), color='black')
#     plt.plot([min(p,t),max(p,t)],[min(p,t),max(p,t)],color='blue')
    plt.plot(np.array([inf,sup]),np.array([inf,sup]),color='red')
    plt.fill_between(np.array([inf,sup]),np.array([inf,sup])-7,np.array([inf,sup])+7,facecolor='mistyrose')
    plt.scatter(t, p, color='black')
    plt.xlabel("Observed {0} date".format(season), fontsize=20)
    plt.xticks(label_locations, date_labels,fontsize=20)
    plt.ylabel("Predicted {0} date".format(season), fontsize=20)
    plt.yticks(label_locations, date_labels,fontsize=20)

    # calculate correlation
    corr = round(pearsonr(p, t)[0], 2)

    # Create list of labels for data points
    labels = []
    for x, y, s in zip(t, p, years):
        k = (x, y)
        if k in years_dict:
            year_list = years_dict.pop(k)
            if k != (0, 0):
                # Format list to string
                labels.append(plt.text(x, y, "[" + ','.join(year_list) + "]"))
            else:
                # If k is (0, 0) then year_list is 'Remaining Years'
                labels.append(plt.text(x, y, year_list))
        else:
            labels.append(plt.text(x, y, ''))

    # use adjust text to ensure labels overlap as little as possible
    adjust_text(labels, only_move={'points': 'xy', 'text': 'xy'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
#     outfile = "{0}_dates_forecasted_at_{1}_days_{2}_by_{3}.png".format(season, lead_day, region_name, model)
    outfile = "{0}_{1}_{2}_{3}".format(season, lead_day, region_name, model)

    savepath = os.path.join(savepath, "evaluations")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    savepath = os.path.join(savepath, "port_correlation")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

#     plt.savefig(os.path.join(savepath, outfile))
#     plt.suptitle(f"Predicted vs observed dates at {region_name}")

    plt.tight_layout()
#     plt.gcf().autofmt_xdate() rotating the xticks

    plt.savefig(os.path.join(savepath, outfile)+'.eps')
#     plt.title("{0} Dates of {1} at\n{2} ({3}N, {4}W)\nfor {5} day forecasts (r = {6}), N = {7}"
#               .format(season, main_region,region_name, location[0], abs(location[1]), lead_day, corr, len(p)))
#     plt.tight_layout()
    plt.savefig(os.path.join(savepath, outfile)+'.png')
    plt.close(plt.gcf())


def evaluate_freezeup_breakup_at_ports(savepath, landmask, region, forecast_days_forward, days_of_historic_input, lead_days,
                                       port_filename, raster_size, lat_bounds, long_bounds, season, load=True):
    """
    :param savepath: str
            path to a directory containing models evaluated at each fold
            e.g /work/IcePresence/NWT_Freeze_v2/H3-F30/model_name
    :param landmask: np.ndarray
            binary 2D array representing land
    :param forecast_days_forward: int
            number of days forecasted forwards
    :param days_of_historic_input: int
            number of days historical days observed by the model
    :param lead_days: int
            the desired lead days to observe - 1. i.e.
            if we wanted 30 day forecasts, we need to input 29
    :param port_filename: string
            name of the yaml file listing ports and their coordinates
    :param raster_size: [int, int]
            spatial resolution of the model
    :param lat_bounds: [float, float]
            latitude bounds of the model
    :param long_bounds: [float, float]
            longitude bound of the model
    :param season: string
            type of season
                - Freeze-up
                - Breakup
    :param load: bool
            if false, code will generate npy files even if they have already been generated
    """

    # calculate correlation at specific locations
    ports_path = resource_filename('sifnet',
                                   os.path.join('medium_term_ice_forecasting', 'support_files', port_filename))

    with open(ports_path, 'r') as f:
        ports = yaml.safe_load(f)

    #plot_these_locs = []
    #for loc in sample_locations:
    #    loc_name = f"Sample point at ({loc[0]},{loc[1]})"

    for name, coord in ports.items():
        for lead_day in lead_days:
            plot_correlation_at_port(savepath, landmask, region, coord, lead_day, forecast_days_forward,
                                         days_of_historic_input, name, raster_size=raster_size,
                                         lat_bounds=lat_bounds, long_bounds=long_bounds,
                                         season=season, model='modeled', load=load)


def evaluate_freezeup_breakup_at_region(savepath, landmask, forecast_days_forward, days_of_historic_input, lead_days=[29],
                              region_name='Hudson Bay', lat_bounds=None, long_bounds=None, season='Freeze-up',
                              load=True):
    """
    Plot all evaluation plots

    :param savepath: str
            path to a directory containing models evaluated at each fold
            e.g /work/IcePresence/NWT_Freeze_v2/H3-F30/model_name
    :param landmask: np.ndarray
            binary 2D array representing land
    :param forecast_days_forward: int
            number of days forecasted forwards
    :param days_of_historic_input: int
            number of days historical days observed by the model
    :param lead_days: list[int]
            list of lead days the model should be evaluated on
    :param region_name: string
            the name of the region (i.e. Hudson Bay, Baffin Bay, NWT)
    :param raster_size: [int, int]
            spatial resolution of the model
    :param lat_bounds: [float, float]
            latitude bounds of the model
    :param long_bounds: [float, float]
            longitude bound of the model
    :param model: [float, float]
            type of model used to make predictions
                - modeled
                - normals
                - persistence
    :param season: string
            type of season
                - Freeze-up
                - Breakup
    :param load: bool
            if false, code will generate npy files even if they have already been generated
    """
    for lead_day in lead_days:
        plot_correlation_map(savepath, landmask, lead_day, forecast_days_forward, days_of_historic_input, lat_bounds,
                             long_bounds, model='modeled', region_name=region_name, season=season, load=load)

        for model in ['modeled', 'normals']:
            plot_accuracy_within_N_days(savepath, landmask, lead_day, 7, forecast_days_forward, days_of_historic_input,
                                        lat_bounds, long_bounds, model=model, season=season, load=load)


def is_same_element(a):
    """
    Check if non-empty list a has the same element.
    Useful to check before calculating correlation

    :param a: List
    :return: bool
        True if a has the same element, False if not
    """
    e = a[0]
    for element in a:
        if element != e:
            return False
    return True


def freezeup_breakup_not_found(p, t, begin, end):
    """
    Check if freeze-up breakup period is not found

    :param p: list[int]
           list of predicted indices
    :param t: list[int]
           list of true indices
    """
    wrap_around = bool(end[0] < begin[0] or (end[0] == begin[0] and end[1] < begin[1]))

    for k in range(len(p)):
        if wrap_around:
            delta = date(k + 1986, end[0], end[1]) \
                    - date(k + 1985, begin[0], begin[1])
        else:
            delta = date(k + 1985, end[0], end[1]) \
                    - date(k + 1985, begin[0], begin[1])
        if p[k] == -1:
            p[k] = delta.days
        if t[k] == -1:
            t[k] = delta.days

