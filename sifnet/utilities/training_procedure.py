"""
Defines a function which trains a single given model and returns the model and its training history.
Also saves the trained model to the provided savepath.

Routine is invariant to which geographic region it is being run on, freeze-up vs breakup, and forecast duration.
Routine is also invariant to the problem at hand, and should also be suitable for domains which we are not
currently testing, such as ice concentration forecasting.

Highly configurable.
Optional features include:
    - multiple GPUs (implemented via MultiGPUModel)
    - Model checkpointing
    - Early stopping
    - Stochastic gradient descent optimizer (Default Adam)
    - Learning rate decay
    - Custom loss functions


Notes
-----
A demo is provided at the bottom of the file for how the function may be used without using the experiment class.
Once upgraded to newer TensorFlow release, update multiGPU to use DistributedStrategy instead of MultiGPUModel.

Execution time
--------------
Highly variable based on model complexity, dataset size, number of epochs, batch size.

References
----------

Examples
--------
See bottom of file.

"""

import os
import time
import traceback

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard

from numba import cuda

from sifnet.data.DatasetManager import DatasetManager
from sifnet.medium_term_ice_forecasting.utilities.model_utilities import custom_objects
import sifnet.utilities.gpu as gpu


def training_procedure(model, model_inputs, model_outputs, dataset_manager, savepath, name, restrict_to_gpu=True,
                       **kwargs):
    """
    Trains a single given model. Returns the model and its training history.
    Also saves the trained model to the provided savepath.

    :param model: tf.keras.models.Model
                    The model to be trained

    :param model_inputs: list of tuples
        List of (generator_function, array_shape, datatype)
        These tuples are the return type of functions from GeneratorFunctions
        e.g. [historical_all_channels(dataset)]

    :param model_outputs: list of tuples
            List of (generator_function, array_shape, datatype)
            These tuples are the return type of functions from GeneratorFunctions
            e.g. [future_single_channel_thresholded(dataset)]

    :param dataset_manager: DatasetManager object
           The datasetmanager used to create train and val generators
           Must be preconfigured.

    :param savepath: string
            Valid path to directory where trained model where be saved

    :param name: string
             Name of the saved model

    :param restrict_to_gpu: bool
            If true, gpu context will be restricted to only 1 GPU. Default True.

    :param kwargs: optional keyword arguments
                    -num_gpu: int                   Number of GPUs to utilize, default 1
                    -max_epochs: int                Maximum number of training epochs, default 50
                    -use_tensorboard: bool          If tensorboard should be enabled, default False
                    -bs_per_gpu: int                Batch size per gpu, default 1
                    -patience: int                  Patience for use with EarlyStopping, default 8.
                    -loss_function: Function        The loss function, default binary_crossentropy
                    -metrics: list of Functions     Metrics to monitor during training.
                                                        default [accuracy, precision, recall]
                    -safe_gen: bool                 If safe_generators should be used. Default True
                    -use_multithreading: bool       If multithreaded generators should be used. Default True
                    -use_checkpoints: bool          If model checkpointing should be used. Default True
                    -lr_decay: float                Learning rate decay rate. Applied each batch. Should be small.
                    -monitor: string                The metric to be monitored for early stopping and checkpointing.
                                                        default 'val_loss'
                    -optimizer: string              Name of the optimizer to use. Default 'Adam'
                    -initial_lr: float              Initial learning rate. Default 0.003

    :return: History object
    """

    # validate mandatory arguments
    # raise error in the upgraded version
    # if type(model) != tf.keras.models.Model:
    #     raise TypeError('Received model of non-tf.keras.models.Model type')

    if type(dataset_manager) != DatasetManager:
        raise TypeError('Received dataset_manager of non-DatasetManager type' +
                        'Instead of type {}'.format(type(dataset_manager)))

    if not dataset_manager.configured:
        raise ValueError('dataset_manager has not been configured')

    if not os.path.exists(savepath):
        raise ValueError('savepath does not exist')

    if type(name) != str:
        raise TypeError('Received name of non-string type')

    # validate keyword arguments
    if 'num_gpu' in kwargs:
        num_gpu = kwargs.pop('num_gpu')
        if type(num_gpu) != int:
            raise TypeError('Received kwarg NUM_GPU of non-int type')
    else:
        num_gpu = 1

    # Number of complete passes over training set
    # Early stopping means full range of epochs rarely executed
    if 'max_epochs' in kwargs:
        max_epochs = kwargs.pop('max_epochs')
        if type(max_epochs) != int:
            raise TypeError('Recieved kwarg max_epochs of non-int type')
    else:
        max_epochs = 50

    if 'use_tensorboard' in kwargs:
        use_tensorboard = kwargs.pop('use_tensorboard')
        if type(use_tensorboard) != bool:
            raise TypeError('Received kwarg use_tensorboard of non-bool type')
    else:
        use_tensorboard = False

    if 'bs_per_gpu' in kwargs:
        bs_per_gpu = kwargs.pop('bs_per_gpu')
        if type(bs_per_gpu) != int:
            raise TypeError('Received kwarg bs_per_gpu of non-int type')
    else:
        bs_per_gpu = 1

    if 'patience' in kwargs:
        patience = kwargs.pop('patience')
        if type(patience) != int:
            raise TypeError('Received kwarg patience of non-int type')
    else:
        patience = 8

    if 'loss_function' in kwargs:
        loss_function = kwargs.pop('loss_function')
    else:
        loss_function = tf.keras.losses.binary_crossentropy

    if 'metrics' in kwargs:
        metrics = kwargs.pop('metrics')
        if type(metrics) != list:
            raise TypeError('Received kwarg metrics of non-list type')
    else:
        metrics = [tf.keras.metrics.binary_accuracy,
                   tf.keras.metrics.Precision(),
                   tf.keras.metrics.Recall()]

    if 'safe_gen' in kwargs:
        safe = kwargs.pop('safe_gen')
        if type(safe) != bool:
            raise TypeError('Received kwarg safe_gen of non-bool type')
    else:
        safe = True

    if 'use_multithreading' in kwargs:
        multithreading = kwargs.pop('use_multithreading')
        if type(multithreading) != bool:
            raise TypeError('Received kwarg safe_gen of non-bool type')
    else:
        multithreading = True

    if 'use_checkpoints' in kwargs:
        use_checkpoints = kwargs.pop('use_checkpoints')
        if type(use_checkpoints) != bool:
            raise TypeError('Received kwarg use_checkpoints of non-bool type')
    else:
        use_checkpoints = True

    if 'use_early_stopping' in kwargs:
        use_early_stopping = kwargs.pop('use_early_stopping')
        if type(use_early_stopping) != bool:
            raise TypeError('Received kwarg use_early_stopping of non-bool type')
    else:
        use_early_stopping = True

    if 'lr_decay' in kwargs:
        lr_decay = kwargs.pop('lr_decay')
        if type(lr_decay) != float:
            raise TypeError('Received kwarg lr_decay of non-float type')
    else:
        lr_decay = 0.

    if 'monitor' in kwargs:
        monitor = kwargs.pop('monitor')
        if type(monitor) != str:
            raise TypeError('Received kwarg monitor of non-string type')
    else:
        monitor = 'val_loss'

    if 'optimizer' in kwargs:
        opt_str = kwargs.pop('optimizer')
        valid_opts = {'Adam', 'SGD'}  # There are more, but these are all supported for now.
        if type(opt_str) != str:
            raise TypeError('Received kwarg optimizer of non-string type')
        if opt_str not in valid_opts:
            raise ValueError('Received kwarg optimizer of value {}. Valid values are {}'.format(opt_str, valid_opts))
    else:
        opt_str = 'Adam'

    if 'initial_lr' in kwargs:
        initial_lr = kwargs.pop('initial_lr')
        if type(initial_lr) != float:
            raise TypeError('Received kwarg initial_lr of non-float type')
        if initial_lr <= 0:
            raise ValueError('Received kwarg initial_lr of non-positive value')
    else:
        initial_lr = 0.003

    if not safe:
        max_q_size = 1
    else:
        max_q_size = 10

    # check for any remaining keyword arguments
    if kwargs:
        raise TypeError('Received unexpected keyword arguments: {}'.format(kwargs))

    if restrict_to_gpu:
        gpu.restrict_to_available_gpu(num_gpu)

    # making batch size too large while setting tensorboard configurations may lead to an
    # out of resource error. If this happens, an easy way to fix this would be to reduce
    # the batch size
    batch_size = bs_per_gpu * num_gpu

    tensorboard_config = dict(
        activation=use_tensorboard,
        histogram_freq=1
    )

    print("Info - Reading in files", flush=True)

    best_metrics = []
    start_time = time.time()
    try:

        print("INFO - Training model")
        print("INFO - Generator safe mode: {}".format(safe))
        print("INFO - Generator Multithreading mode: {}".format(multithreading))

        (train_gen, train_steps_per_epoch) = dataset_manager.make_generator(batch_size, 'train',
                                                                            input_functions=model_inputs,
                                                                            output_functions=model_outputs,
                                                                            multithreading=multithreading,
                                                                            safe=safe)

        (val_gen, val_steps_per_epoch) = dataset_manager.make_generator(batch_size, 'val',
                                                                        input_functions=model_inputs,
                                                                        output_functions=model_outputs,
                                                                        multithreading=multithreading,
                                                                        safe=safe)

        if val_steps_per_epoch == 0:
            print('INFO: NOT USING VALIDATION GENERATOR!')
            val_gen = None

        # create multi_gpu_model if more than 1 gpu specified
        if num_gpu > 1:
            train_model = keras.utils.multi_gpu_model(model, num_gpu, cpu_merge=False, cpu_relocation=True)
        else:
            train_model = model

        if opt_str == 'Adam':
            opt = tf.keras.optimizers.Adam(initial_lr, decay=lr_decay, clipnorm=0.1)
        elif opt_str == 'SGD':
            print('INFO: USING SGD OPTIMIZER')
            opt = tf.keras.optimizers.SGD(initial_lr, nesterov=True, decay=lr_decay, clipnorm=0.1, momentum=0.9)

        train_model.compile(opt,
                            loss=loss_function,
                            metrics=metrics)

        callbacks = []

        if use_early_stopping:
            # Monitors a metric. Default val_loss,
            early = tf.keras.callbacks.EarlyStopping(patience=patience, monitor=monitor)
            callbacks.append(early)

        if use_checkpoints:
            # Monitors a metric. Default val_loss,
            checkpoint_name = 'checkpoint.h5'
            checkpoint_path = os.path.join(savepath, checkpoint_name)
            checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor=monitor, verbose=1,
                                                            save_best_only=True)

            callbacks.append(checkpoint)

        # If tensorboard activation is true, include tensorboard to list of callbacks to be passed to fit_generator
        if tensorboard_config["activation"]:
            log_name = 'Training_log'
            log_output_path = os.path.join(savepath, "tensorboard", log_name)
            print('Tensorboard logging at {}'.format(log_output_path))

            tensorboard = TensorBoard(log_dir=log_output_path)
            callbacks.append(tensorboard)

        if val_steps_per_epoch > 0:
            history = train_model.fit(train_gen, steps_per_epoch=train_steps_per_epoch,
                                            epochs=max_epochs,
                                            validation_data=val_gen, validation_steps=val_steps_per_epoch,
                                            callbacks=callbacks, max_queue_size=max_q_size)
        else:
            history = train_model.fit(train_gen, steps_per_epoch=train_steps_per_epoch,
                                            epochs=max_epochs,
                                            callbacks=callbacks, max_queue_size=max_q_size)

        if use_checkpoints:
            train_model = tf.keras.models.load_model(checkpoint_path, custom_objects=custom_objects)

        if num_gpu > 1:
            # Second last 'layer' of multi_gpu_model is the original model.
            model.set_weights(train_model.layers[-2].get_weights())
            model.compile(optimizer=train_model.optimizer,
                          loss=loss_function,
                          metrics=metrics)
        else:
            model = train_model

        print("INFO - Saving Model")
        model_path = os.path.join(savepath, "{}.h5".format(name))
        model.save(model_path)  # save the non-multi-gpu version
        model = keras.models.load_model(model_path)  # bugfix?

        if use_checkpoints:
            # only remove checkpoint after model has saved successfully.
            os.remove(checkpoint_path)

        print("Info - Time elapsed")
        total_time = time.time()-start_time
        print("--- %s seconds ---" % (total_time))
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        if restrict_to_gpu:
            cuda.close()
        raise Exception('Encountered an internal exception. See above message.')

    finally:
        if restrict_to_gpu:
            cuda.close()

    return history, model, total_time


if __name__ == '__main__':
    """
    A demonstration of the use of training_procedure outside of the experiment pipeline
    """
    import os
    import yaml
    from pkg_resources import resource_filename
    import sifnet.medium_term_ice_forecasting.ice_presence.model as models
    import sifnet.medium_term_ice_forecasting.ice_presence.future_channel_model as f_models
    from sifnet.data.GeneratorFunctions import historical_all_channels, future_single_channel_thresholded, \
        future_multi_channel
    import matplotlib.pyplot as plt

    # load a dataset
    with open(resource_filename('sifnet', 'medium_term_ice_forecasting/datasets/Hudson_Freeze_v2.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        dsm = DatasetManager(config)

    # configure our DatasetManager
    # dsm.config(3, 30, validation_years=[1989, 2000, 2002, 2014, 2016], test_years=[1990, 2003, 2012, 2013, 2017])
    dsm.config(3, 30, validation_years=[2017], test_years=[])
    print(dsm.train_dates)
    print(dsm.val_dates)
    print(dsm.test_dates)

    # prepare our inputs & outputs
    # inputs = [historical_all_channels(dsm)] # for leaky_baseline_30_day_forecast model
    inputs = [historical_all_channels(dsm), future_multi_channel(dsm, [2, 4, 5])]  # t2m , u10, v10 future inputs
    outputs = [future_single_channel_thresholded(dsm)]

    savepath = '/work/training_demo'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # build a model - this is the model to be trained.
    # model = models.leaky_baseline_30_day_forecast()
    # model_name = 'training_demo_model'

    model = f_models.leaky_baseline_30_day_forecast_1t_fc()  # default input shape
    model_name = 'augmented_demo_model'

    # create our keyword arguments
    # kwargs = dict(max_epochs=60, use_early_stopping=False, use_checkpoints=False, bs_per_gpu=13, num_gpu=2)
    kwargs = dict(max_epochs=60, use_early_stopping=False, use_checkpoints=False, bs_per_gpu=8, num_gpu=2)

    # finally, call train. Returned is training history and the trained model.
    # trained model is also saved at savepath/model_name.h5
    history, model = training_procedure(model, inputs, outputs, dsm, savepath, model_name, **kwargs)

    # plot the training history
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(history.history['val_loss'], label='val loss')
    ax1.plot(history.history['loss'], label='train loss')
    ax1.legend(loc='upper right')
    ax1.grid()
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    fig.savefig('/work/training_demo/'+model_name+'Loss.png')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(history.history['val_binary_accuracy'], label='val accuracy')
    ax1.plot(history.history['binary_accuracy'], label='train accuracy')
    ax1.legend(loc='lower right')
    ax1.set_title('Accuracy')
    ax1.grid()
    ax1.set_xlabel('Epochs')
    fig.savefig('/work/training_demo/'+model_name+'Accuracy.png')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(history.history['val_precision'], label='val precision')
    ax1.plot(history.history['precision'], label='train precision')
    ax1.legend(loc='lower right')
    ax1.set_title('Precision')
    ax1.grid()
    ax1.set_xlabel('Epochs')
    fig.savefig('/work/training_demo/'+model_name+'Precision.png')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(history.history['val_recall'], label='val recall')
    ax1.plot(history.history['recall'], label='train recall')
    ax1.legend(loc='lower right')
    ax1.set_title('Recall')
    ax1.grid()
    ax1.set_xlabel('Epochs')
    fig.savefig('/work/training_demo/'+model_name+'Recall.png')
