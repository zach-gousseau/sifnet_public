"""
A set of standardized evaluation functions. These functions are used by the standard evaluation utilities but
are also suitable for independent use.

They return metrics on a per-day basis and a forecast-wide average.

Written by Matthew King
February, 2019
@ NRC
"""

import numpy as np


# expecting n-d arrays (samples, days, lat, lon)
# calculates the binary accuracy, thresholding the predictions at 0.5
# Average is taken across samples so average accuracy by day.
def np_accuracy(y_true, y_predict, mask=None):
    """
    A function which compute the per-day and average forecast accuracy. Besides calculating the
    accuracy of y_predict compared to y_true, the function allows user to take in a binary mask
    that filters out certain values during comparison.

    1s in the mask usually represent land, and 0s in the mask represent bodies of water.

    :param y_true: Numpy array of forecast base truths. Numpy array, shape=(510, 30, 160, 300)
    :param y_predict: Numpy array of model forecasts. Numpy array. shape=(510, 30, 160,300)
    :param mask: land_mask. Hudson_Freeze[0] is the loaded config from _get_hudson_freeze
    :return: tuple (average accuracy, daily accuracy)
    """
    N = y_true.size
    if N == 0:
        return -1

    # y_predict is expressed as a probability. Convert to binary.
    y_predict = np.array(y_predict >= 0.5, dtype=np.uint8)
    # check which predictions are correct
    correct = np.array(np.equal(y_true, y_predict)).astype(np.uint8)
    del y_predict

    if mask is not None:
        print('INFO: np_accuracy: Using landmask with {} land cells'.format(np.sum(mask)))
        inverted_mask = (1 - mask).astype(np.uint8)
        # Find total number of grid locations that are not land

        # apply mask
        correct *= inverted_mask
        total = np.count_nonzero(inverted_mask == 1)
        del inverted_mask
    else:
        total = y_true[0, 0].size

    # find daily accuracy
    # sum over height and width and divide by the total maximum possible number of non-land cells.
    daily_breakdown = np.mean(np.sum(np.sum(correct, axis=-1), axis=-1) / total, axis=0)

    overall = np.mean(daily_breakdown)  # average over forecast duration

    return overall, daily_breakdown


# expecting n-d arrays (samples, days, lat, lon)
# thresholds predictions at 0.5
# returns the precision = (True Positives) / (All Predicted Positives)
# taken as the mean as across all images.
def np_precision(y_true, y_predict, mask=None):
    """
    A function which compute the per-day and average forecast precision. Besides calculating the
    precision of y_predict compared to y_true, the function allows user to take in a binary mask
    that filters out certain values during comparison.

    1s in the mask usually represent land, and 0s in the mask represent bodies of water.

    Where precision = true_positives / all_predicted_positives

    :param y_true: Numpy array of forecast base truths. Numpy array, shape=(510, 30, 160, 300)
    :param y_predict: Numpy array of model forecasts. Numpy array. shape=(510, 30, 160,300)
    :param mask: land_mask
    :return: tuple (average precision, daily precision)
    """

    y_predict = np.array(y_predict >= 0.5, dtype=np.uint8)

    true_positives = np.multiply(y_true, y_predict).astype(np.uint8)
    # apply mask to true positives element wise
    if mask is not None:
        print('INFO: np_precision: Using landmask with {} land cells'.format(np.sum(mask)))
        inverted_mask = (1 - mask).astype(np.uint8)
        true_positives *= inverted_mask

        # apply mask to all predictions element wise
        y_predict *= inverted_mask
        del inverted_mask

    daily_true_positives = np.sum(np.sum(true_positives, axis=-1), axis=-1)  # sum over height and width

    # all positives are all predicted (forecasted) positives
    daily_all_positives = np.sum(np.sum(y_predict, axis=-1), axis=-1)

    # daily precision
    daily = np.mean(np.divide(daily_true_positives, daily_all_positives + 0.000000001), axis=0)  # P = TP / PP

    # mean precision
    overall = np.mean(daily)

    return overall, daily


# expecting n-d arrays (samples, days, lat, lon)
# rounds predicts at 0.5 threshold
# returns recall = (True Positives) / (All Desired Positives)
# taken as the mean as across all images.
def np_recall(y_true, y_predict, mask=None):
    """
    A function which compute the per-day and average forecast recall.  Besides calculating the
    recall of y_predict compared to y_true, the function allows user to take in a binary mask
    that filters out certain values during comparison.

    1s in the mask usually represent land, and 0s in the mask represent bodies of water.

    Where recall = true_positives / all_baseline_positives

    :param y_true: Numpy array of forecast base truths. Numpy array, shape=(510, 30, 160, 300)
    :param y_predict: Numpy array of model forecasts. Numpy array. shape=(510, 30, 160,300)
    :param mask: land_mask
    :return: tuple (average precision, daily precision)
    """
    y_predict = np.array(y_predict >= 0.5, dtype=np.uint8)
    true_positives = np.multiply(y_true, y_predict).astype(np.uint8)
    del y_predict

    if mask is not None:
        print('INFO: np_recall: Using landmask with {} land cells'.format(np.sum(mask)))
        inverted_mask = (1 - mask).astype(np.uint8)

        true_positives *= inverted_mask
        y_true *= inverted_mask

    daily_true_positives = np.sum(np.sum(true_positives, axis=-1), axis=-1)  # sum over height and width

    # base positives are all positives from the base truth
    daily_base_positives = np.sum(np.sum(y_true, axis=-1), axis=-1)  # sum over height and width

    # divide each day's true positives by base_positives. Then take mean over all forecasts
    daily = np.mean(np.divide(daily_true_positives, daily_base_positives+0.000000001), axis=0) # R = TP / AP

    # mean recall
    overall = np.mean(daily)  # mean over forecast duration

    return overall, daily

def grid_f1_score(y_true,y_predict,mask=None):
    """
    A function which compute the f1_score in the [sample x days] dimension. Besides calculating the
    recall of y_predict compared to y_true, the function allows user to take in a binary mask
    that filters out certain values during comparison.

    1s in the mask usually represent land, and 0s in the mask represent bodies of water.

    Where f1-score = 2*precision*recall / (precision+recall)

    :param y_true: Numpy array of forecast base truths. Numpy array, shape=(510, 30, 160, 300)
    :param y_predict: Numpy array of model forecasts. Numpy array. shape=(510, 30, 160,300)
    :param mask: land_mask
    :return: Numpy array of f1-scores. Numpy array, shape=(510, 30)
    """
    y_predict = np.array(y_predict >= 0.5, dtype=np.uint8)
    true_positives = np.multiply(y_true, y_predict).astype(np.uint8)
#     del y_predict

    if mask is not None:
        print('INFO: np_recall: Using landmask with {} land cells'.format(np.sum(mask)))
        inverted_mask = (1 - mask).astype(np.uint8)
        true_positives *= inverted_mask
        y_true *= inverted_mask
        y_predict *= inverted_mask
        total = np.count_nonzero(inverted_mask == 1)
        del inverted_mask

    daily_true_positives = np.sum(np.sum(true_positives, axis=-1), axis=-1)  # sum over height and width

    # base positives are all positives from the base truth
    daily_base_positives = np.sum(np.sum(y_true, axis=-1), axis=-1)  # sum over height and width
    # divide each day's true positives by base_positives. Then take mean over all forecasts
    recall = np.divide(daily_true_positives, daily_base_positives+0.000000001) # R = TP / AP   

    # all positives are all predicted (forecasted) positives
    daily_all_positives = np.sum(np.sum(y_predict, axis=-1), axis=-1)
    # daily precision
    precision = np.divide(daily_true_positives, daily_all_positives + 0.000000001) # P = TP / PP
    
    f1_score = 2*precision*recall/(precision+recall+0.000000001)
    
    return f1_score

def np_f1_score(y_true, y_predict, mask=None):
    """
    A function which compute the per-day and average forecast f1_score weighted by class samples. Besides calculating the
    recall of y_predict compared to y_true, the function allows user to take in a binary mask
    that filters out certain values during comparison.

    1s in the mask usually represent land, and 0s in the mask represent bodies of water.

    Where f1-score = (positive_samples*f1_score_positives + negative_samples*f1_score_negatives) / total_samples

    :param y_true: Numpy array of forecast base truths. Numpy array, shape=(510, 30, 160, 300)
    :param y_predict: Numpy array of model forecasts. Numpy array. shape=(510, 30, 160,300)
    :param mask: land_mask
    :return: tuple (average f1_socre, daily f1_score)
    """
    
    f1_score_pos = grid_f1_score(y_true,y_predict,mask) 
    f1_score_neg = grid_f1_score(1-y_true,1-y_predict,mask) 
    
    if mask is not None:
        inverted_mask = (1 - mask).astype(np.uint8)
        total = np.count_nonzero(inverted_mask == 1)
        y_true *= inverted_mask
        del inverted_mask
        
    daily_base_positives = np.sum(np.sum(y_true, axis=-1), axis=-1)
    daily_base_negatives = total-daily_base_positives#np.sum(np.sum(1-y_true, axis=-1), axis=-1)
    
    f1_score = (daily_base_positives*f1_score_pos + daily_base_negatives*f1_score_neg)/total
    
    daily = np.mean(f1_score, axis=0)  
    # mean precision
    overall = np.mean(daily)

    return overall, daily

def np_brier(y_true, y_predict, mask=None):
    """
    A function which compute the per-day and average brier score. 

    1s in the mask usually represent land, and 0s in the mask represent bodies of water.

    :param y_true: Numpy array of forecast base truths. Numpy array, shape=(510, 30, 160, 300)
    :param y_predict: Numpy array of model forecasts. Numpy array. shape=(510, 30, 160,300)
    :param mask: land_mask. Hudson_Freeze[0] is the loaded config from _get_hudson_freeze
    :return: tuple (average score, daily score)
    """
    N = y_true.size
    if N == 0:
        return -1

    score = (y_predict-y_true)**2
    del y_predict

    if mask is not None:
        print('INFO: np_accuracy: Using landmask with {} land cells'.format(np.sum(mask)))
        inverted_mask = (1 - mask).astype(np.uint8)
        # Find total number of grid locations that are not land

        # apply mask
        score *= inverted_mask
        total = np.count_nonzero(inverted_mask == 1)
        del inverted_mask
    else:
        total = y_true[0, 0].size

    # find daily accuracy
    # sum over height and width and divide by the total maximum possible number of non-land cells.
    daily_breakdown = np.mean(np.sum(np.sum(score, axis=-1), axis=-1) / total, axis=0)

    overall = np.mean(daily_breakdown)  # average over forecast duration

    return overall, daily_breakdown


def confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1)).ravel()

    return tp, fp, fn, tn

def np_accuracy_map(y_true, y_predict):
    """
    A function which compute the per-day and average forecast accuracy as a function of spatial location.
    Locations which are part of the land-mask are expected to have 100% accuracy.

    :param y_true: Numpy array of forecast base truths. Numpy array, shape=(510, 30, 160, 300)
    :param y_predict: Numpy array of model forecasts. Numpy array. shape=(510, 30, 160,300)
    :return: tuple (accuracy_map, daily_accuracy_map)
    """

    # y_predict is expressed as a probability. Convert to binary.
    y_predict = np.array(y_predict > 0.5, dtype=np.uint8)
    # check which predictions are correct
    correct = np.array(np.equal(y_true, y_predict)).astype(np.uint8)
    del y_predict
    daily_accuracy = np.mean(correct, axis=0)  # (30, 160, 300)
    del correct
    overall_accuracy = np.mean(daily_accuracy, axis=0)  # (160, 300)

    return overall_accuracy, daily_accuracy


def np_bias_map(y_true, y_predict):
    """
    A function which compute the per-day and average forecast bias as a function of spatial location.

    :param y_true: Numpy array of forecast base truths. Numpy array, shape=(510, 30, 160, 300)
    :param y_predict: Numpy array of model forecasts. Numpy array. shape=(510, 30, 160,300)
    :return: tuple (bias_map, daily_bias_map)
    """

    # y_predict is expressed as a probability. Convert to binary.
    y_predict = np.array(y_predict > 0.5, dtype=np.int8)
    # check which predictions are correct
    bias = y_predict - y_true
    del y_predict
    daily_bias = np.mean(bias, axis=0)  # (30, 160, 300)
    del bias
    overall_bias = np.mean(daily_bias, axis=0)  # (160, 300)

    return overall_bias, daily_bias

def np_prob_map(y_true, y_predict):
    """
    A function which compute the per-day and average forecast bias as a function of spatial location.

    :param y_true: Numpy array of forecast base truths. Numpy array, shape=(510, 30, 160, 300)
    :param y_predict: Numpy array of model forecasts. Numpy array. shape=(510, 30, 160,300)
    :return: tuple (bias_map, daily_bias_map)
    """

    # y_predict is expressed as a probability. Convert to binary.
#     y_predict = np.array(y_predict > 0.5, dtype=np.int8)
    # check which predictions are correct
    d_prob = y_predict.astype(np.float32) - y_true.astype(np.float32)
    del y_predict
    daily_d_prob = np.mean(d_prob, axis=0)  # (30, 160, 300)
    del d_prob
    overall_d_prob = np.mean(daily_d_prob, axis=0)  # (160, 300)

    return overall_d_prob, daily_d_prob

def main():
    y_true = np.array([[[1, 0, 1], [0, 0, 1], [1, 1, 1]],
                       [[1, 0, 0], [0, 1, 1], [1, 1, 1]]])

    y_true = np.array([y_true, y_true])  # 2 samples, identical

    y_predict = np.array([[[0.8, 0.55, 0.6], [0.125, 0.45, 0.55], [0.75, 0.7, 0.8]],
                          [[0.9, 0.65, 0.25], [0.1, 0.6, 0.9], [0.4, 0.9, 0.6]]])

    y_predict = np.array([y_predict, y_predict])

    y_predict_binary = np.array([[[1, 1, 1], [0, 0, 1], [1, 1, 1]],
                                 [[1, 1, 0], [0, 1, 1], [0, 1, 1]]])

    y_predict_binary = np.array([y_predict_binary, y_predict_binary])
    print(confusion_matrix(y_true, y_predict_binary))

    mask = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])

    print(y_true)
    print(y_predict)

    # print(y_true.shape)  #(2,2,3,3)
    # print(y_predict.shape) #(2,2,3,3)

    print(np_accuracy(y_true, y_predict, mask))  # 15/18, [8/9, 7/9] if mask is all 0s

    print(np_precision(y_true, y_predict, mask))  # 11/13, [6/7, 5/6]

    print(np_recall(y_true, y_predict, mask))  # 11/12, [1, 5/6]

    # The following should be the output if mask is applied
    # (0.8125, array([0.875, 0.75]))
    # (0.816666666, array([0.83333333, 0.8]))
    # (0.9, array([1., 0.8]))


if __name__ == "__main__":
    main()
