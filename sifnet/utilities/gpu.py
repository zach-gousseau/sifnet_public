"""
Functions related to GPU information

"""

import os
import GPUtil
import tensorflow as tf
# from tensorflow.keras.backend import set_session
from tensorflow.compat.v1.keras.backend import set_session

def set_gpu_options(kind="growth", gpu_memory_fraction=1):
    """
    Set GPU options for session. Good practice since we are sharing GPUs. By default,
    keras session take up GPU memory completely
    :param kind: "growth", "fraction"
        Set memory usage as dynamic or with a hard limit
    :param gpu_memory_fraction: (0, 1]
        Percentage of GPU memory we limit the process to. Only used if kind="fraction"
    Provides information related to GPU i.e. memory, number of GPU's, addresses

    :return:
    """
    if kind == "growth":
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
    elif kind == "fraction":
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        config = tf.ConfigProto(gpu_options=gpu_options)

    sess = tf.compat.v1.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


def return_num_available_gpu(limit=4, maxLoad=0.5, maxMemory=0.5):
    """
    Finds number of available GPUs

    :param limit: Integer
        maximum number of GPUs we want to find
    :param maxLoad: [0, 1]
        Maximum load for a GPU to be considered available. Expressed as a percent.
    :param maxMemory: [0, 1]
        Maximum memory usage for a GPU to be considered available. Expressed as a percent.
    :return: int
        number of available GPUs
    """
    return len(GPUtil.getAvailable(limit=limit, maxLoad=maxLoad, maxMemory=maxMemory))


def restrict_to_available_gpu(num_requested_gpus, limit=4, maxLoad=0.5, maxMemory=0.5):
    """
    Restrict access to GPUs that are considered busy. Decides if a a GPU is busy by the
    parameters passed to the function.

    DO NOT USE THIS FUNCTION WITH set_gpu_options. If you do, this function will not work.

    :param num_requested_gpus: Integer
        number of GPUs requested by the user
    :param limit: Integer
        maximum number of GPUs we want to find
    :param maxLoad: [0, 1]
        Maximum load for a GPU to be considered available. Expressed as a percent. 
    :param maxMemory: [0, 1]
        Maximum memory usage for a GPU to be considered available. Expressed as a percent.
    :return:


    """
    # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # limit is the max number of GPUs returned
    # maxload is the maximum load that makes a GPU considered "available"
    # max
    DEVICE_ID_LIST = GPUtil.getAvailable(limit=limit, maxLoad=maxLoad, maxMemory=maxMemory)

    num_available_gpus = len(DEVICE_ID_LIST)
    if num_requested_gpus > num_available_gpus:
        raise Exception("Number of requested GPUs: {0}, number of available GPUs: {1}".format(num_requested_gpus, num_available_gpus))

    device_ids = ','.join(map(str, DEVICE_ID_LIST))

    # restrict device to following GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
