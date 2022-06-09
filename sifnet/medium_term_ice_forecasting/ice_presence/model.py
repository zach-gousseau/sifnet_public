"""
Tensorflow model(s)

The Keras functional API is leveraged.
"""

import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.layers import LeakyReLU

import sifnet.medium_term_ice_forecasting.utilities.model_utilities as mu


def baseline_30_day_forecast(**kwargs):
    """
    Creates a model deep CNN for sea ice forecasting, using the Keras functional API.
    Original model produced spring of 2019.
    Three computational 'towers' encode features based on historical input.
        Tower structure is loosely inspired by inceptionNet architecture.
    Low level features are extrapolated out to 30 timesteps using
    3D transpose convoltional layers.
    The 30-days of features are further processed through a Conv-LSTM.
    A skip connection brings the extrapolated features forward and concatenates them
        with the features produced via the LSTM.
    Finally, the features are integrated into a final per-pixel prediction
        using network-in-a-network structure implemented through time-distributed
        2D Conv layers with (1x1) receptive fields.

    Obsolete architecture.

    :return: Keras model object
    """

    if 'l2reg' in kwargs:
        l2 = kwargs['l2reg']
    else:
        l2 = 0.001

    if 'input_shape' in kwargs:
        input_shape = kwargs['input_shape']
    else:
        input_shape = (3, 160, 300, 8)

    inputs = tf.keras.Input(shape=input_shape)

    # == tower 1

    pad_latlon = kl.ZeroPadding3D(padding=(0, 1, 1), data_format="channels_last",
                                  name="t1p1")(inputs)

    x_simple = kl.Conv3D(12, (2, 3, 3), padding="valid", name="t1_down1",
                         activation="relu",
                         kernel_regularizer=tf.keras.regularizers.l2(l2))(
        pad_latlon)

    pad_latlon = kl.ZeroPadding3D(padding=(0, 2, 2), data_format="channels_last",
                                  name="t1p2", )(x_simple)

    t1 = kl.Conv3D(16, (2, 5, 5), padding="valid", name="t1_down2",
                   activation="relu",
                   kernel_regularizer=tf.keras.regularizers.l2(l2))(
        pad_latlon)

    # == tower 2

    x_dilated = kl.TimeDistributed(
        kl.Conv2D(6, (3, 3), dilation_rate=(5, 5), padding="same",
                  activation="relu",
                  kernel_regularizer=tf.keras.regularizers.l2(l2)),
        name="dilated")(inputs)

    x_lstm = kl.ConvLSTM2D(3, (2, 2), dilation_rate=(3, 3), padding="same",
                           activation="relu",
                           name="LSTM-1", return_sequences=True,
                           kernel_regularizer=tf.keras.regularizers.l2(l2))(
        inputs)

    big = kl.concatenate([inputs, x_dilated, x_lstm], axis=-1)

    pad = kl.ZeroPadding3D(padding=(0, 1, 1), data_format="channels_last",
                           name="t2p1")(big)

    d1 = kl.Conv3D(12, (2, 3, 3), padding="valid", name="t2_d1",
                   activation="relu",
                   kernel_regularizer=tf.keras.regularizers.l2(l2))(pad)

    pad = kl.ZeroPadding3D(padding=(0, 2, 2), data_format="channels_last",
                           name="t2p2")(d1)

    t2 = kl.Conv3D(16, (2, 5, 5), padding="valid", name="t2_d2",
                   activation="relu",
                   kernel_regularizer=tf.keras.regularizers.l2(l2))(pad)

    # == tower 3

    paddeder = kl.ZeroPadding3D(padding=(0, 3, 3), name="t3_pad",
                                data_format="channels_last")(inputs)

    t3 = kl.Conv3D(16, (3, 7, 7), padding="valid", name="tower3",
                   activation="relu",
                   kernel_regularizer=tf.keras.regularizers.l2(l2))(paddeder)

    # Above the towers

    bigdata = kl.concatenate([t1, t2, t3], name="concat_all", axis=-1)

    # Conv3DTranpose transform data out into 30 days

    bigdata = kl.Conv3DTranspose(32, (6, 3, 3), padding="valid",
                                 activation="relu", name="up1",
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                    l2))(bigdata)

    bigdata = kl.Conv3DTranspose(24, (5, 3, 3), strides=(2, 1, 1),
                                 padding="valid", activation="relu", name="up2",
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                    l2))(bigdata)

    bigdata = kl.Conv3DTranspose(16, (2, 3, 3), strides=(2, 1, 1),
                                 padding="valid", activation="relu", name="up3",
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                    l2))(bigdata)

    pre = kl.TimeDistributed(kl.Conv2D(8, (5, 5), padding="valid",
                                       activation="relu",
                                       kernel_regularizer=tf.keras.regularizers.l2(
                                         l2)), name="preLSTM")(bigdata)

    # pre = l.TimeDistributed(l.Conv2D(4,(5,5), padding="valid",
    # activation="relu", name="preLSTM"))(pre)

    LSTM = kl.ConvLSTM2D(4, (3, 3), padding="valid", activation="tanh",
                         name="LSTM-2",
                         return_sequences=True, recurrent_activation="tanh",
                         kernel_regularizer=tf.keras.regularizers.l2(l2))(pre)

    crop = kl.TimeDistributed(kl.Cropping2D(cropping=((3, 3), (3, 3))))(bigdata)

    con = kl.concatenate([LSTM, crop], name="last_concat", axis=-1)

    # NIN
    nin = kl.TimeDistributed(
        kl.Conv2D(40, (1, 1), padding="same", activation="relu",
                  kernel_regularizer=tf.keras.regularizers.l2(l2)),
        name="NIN1")(con)

    nin = kl.TimeDistributed(
        kl.Conv2D(25, (1, 1), padding="same", activation="relu",
                  kernel_regularizer=tf.keras.regularizers.l2(l2)),
        name="NIN2")(nin)

    nin = kl.TimeDistributed(
        kl.Conv2D(15, (1, 1), padding="same", activation="relu",
                  kernel_regularizer=tf.keras.regularizers.l2(l2)),
        name="NIN3")(nin)

    nin = kl.TimeDistributed(
        kl.Conv2D(5, (1, 1), padding="same", activation="relu",
                  kernel_regularizer=tf.keras.regularizers.l2(l2)),
        name="NIN4")(nin)

    nin = kl.TimeDistributed(
        kl.Conv2D(2, (1, 1), padding="same", activation="relu",
                  kernel_regularizer=tf.keras.regularizers.l2(l2)),
        name="NIN5")(nin)

    out = kl.TimeDistributed(
        kl.Conv2D(1, (1, 1), padding="same", activation="sigmoid",
                  kernel_regularizer=tf.keras.regularizers.l2(l2)),
        name="out")(nin)

    # END MODEL #########################################################

    model = tf.keras.Model(inputs=inputs, outputs=out)

    return model


def leaky_baseline_30_day_forecast(**kwargs):
    """
    Creates a model deep CNN for sea ice forecasting, using the Keras functional API.
    Original model produced spring of 2019.
    Three computational 'towers' encode features based on historical input.
        Tower structure is loosely inspired by inceptionNet architecture.
    Low level features are extrapolated out to 30 timesteps using
    3D transpose convoltional layers.
    The 30-days of features are further processed through a Conv-LSTM.
    A skip connection brings the extrapolated features forward and concatenates them
        with the features produced via the LSTM.
    Finally, the features are integrated into a final per-pixel prediction
        using network-in-a-network structure implemented through time-distributed
        2D Conv layers with (1x1) receptive fields.

    Mostly obsolete. Viable for 30 day ensemble.

    :return: Keras model object
    """
    if 'l2reg' in kwargs:
        l2 = kwargs['l2reg']
    else:
        l2 = 0.001

    if 'input_shape' in kwargs:
        input_shape = kwargs['input_shape']
    else:
        input_shape = (3, 160, 300, 8)

    inputs = tf.keras.Input(shape=input_shape)

    # == tower 1

    pad_latlon = kl.ZeroPadding3D(padding=(0, 1, 1), data_format="channels_last",
                                  name="t1p1")(inputs)

    x_simple = kl.Conv3D(12, (2, 3, 3), padding="valid", name="t1_down1",
                         activation="linear",
                         kernel_regularizer=tf.keras.regularizers.l2(l2))(
        pad_latlon)
    x_simple = LeakyReLU()(x_simple)

    pad_latlon = kl.ZeroPadding3D(padding=(0, 2, 2), data_format="channels_last",
                                  name="t1p2", )(x_simple)

    t1 = kl.Conv3D(16, (2, 5, 5), padding="valid", name="t1_down2",
                   activation="linear",
                   kernel_regularizer=tf.keras.regularizers.l2(l2))(
        pad_latlon)

    t1 = LeakyReLU()(t1)

    # == tower 2

    x_dilated = kl.TimeDistributed(
        kl.Conv2D(6, (3, 3), dilation_rate=(5, 5), padding="same",
                  activation="linear",
                  kernel_regularizer=tf.keras.regularizers.l2(l2)),
        name="dilated")(inputs)
    x_dilated = LeakyReLU()(x_dilated)

    x_lstm = kl.ConvLSTM2D(3, (2, 2), dilation_rate=(3, 3), padding="same",
                           activation="relu",
                           name="LSTM-1", return_sequences=True,
                           kernel_regularizer=tf.keras.regularizers.l2(l2))(
        inputs)

    big = kl.concatenate([inputs, x_dilated, x_lstm], axis=-1)

    pad = kl.ZeroPadding3D(padding=(0, 1, 1), data_format="channels_last",
                           name="t2p1")(big)

    d1 = kl.Conv3D(12, (2, 3, 3), padding="valid", name="t2_d1",
                   activation="linear",
                   kernel_regularizer=tf.keras.regularizers.l2(l2))(pad)
    d1 = LeakyReLU()(d1)

    pad = kl.ZeroPadding3D(padding=(0, 2, 2), data_format="channels_last",
                           name="t2p2")(d1)

    t2 = kl.Conv3D(16, (2, 5, 5), padding="valid", name="t2_d2",
                   activation="linear",
                   kernel_regularizer=tf.keras.regularizers.l2(l2))(pad)
    t2 = LeakyReLU()(t2)

    # == tower 3

    paddeder = kl.ZeroPadding3D(padding=(0, 3, 3), name="t3_pad",
                                data_format="channels_last")(inputs)

    t3 = kl.Conv3D(16, (3, 7, 7), padding="valid", name="tower3",
                   activation="linear",
                   kernel_regularizer=tf.keras.regularizers.l2(l2))(paddeder)
    t3 = LeakyReLU()(t3)

    # Above the towers

    bigdata = kl.concatenate([t1, t2, t3], name="concat_all", axis=-1)

    # Conv3DTranpose transform data out into 30 days

    bigdata = kl.Conv3DTranspose(32, (6, 3, 3), padding="valid",
                                 activation="linear", name="up1",
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                    l2))(bigdata)
    bigdata = LeakyReLU()(bigdata)

    bigdata = kl.Conv3DTranspose(24, (5, 3, 3), strides=(2, 1, 1),
                                 padding="valid", activation="linear", name="up2",
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                    l2))(bigdata)
    bigdata = LeakyReLU()(bigdata)

    bigdata = kl.Conv3DTranspose(16, (2, 3, 3), strides=(2, 1, 1),
                                 padding="valid", activation="linear", name="up3",
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                    l2))(bigdata)
    bigdata = LeakyReLU()(bigdata)

    pre = kl.TimeDistributed(kl.Conv2D(8, (5, 5), padding="valid",
                                       activation="linear",
                                       kernel_regularizer=tf.keras.regularizers.l2(
                                         l2)), name="preLSTM")(bigdata)
    pre = LeakyReLU()(pre)


    LSTM = kl.ConvLSTM2D(4, (3, 3), padding="valid", activation="tanh",
                         name="LSTM-2",
                         return_sequences=True, recurrent_activation="tanh",
                         kernel_regularizer=tf.keras.regularizers.l2(l2))(pre)

    crop = kl.TimeDistributed(kl.Cropping2D(cropping=((3, 3), (3, 3))))(bigdata)

    con = kl.concatenate([LSTM, crop], name="last_concat", axis=-1)

    # NIN
    nin = kl.TimeDistributed(
        kl.Conv2D(40, (1, 1), padding="same", activation="linear",
                  kernel_regularizer=tf.keras.regularizers.l2(l2)),
        name="NIN1")(con)
    nin = LeakyReLU()(nin)

    nin = kl.TimeDistributed(
        kl.Conv2D(25, (1, 1), padding="same", activation="linear",
                  kernel_regularizer=tf.keras.regularizers.l2(l2)),
        name="NIN2")(nin)
    nin = LeakyReLU()(nin)

    nin = kl.TimeDistributed(
        kl.Conv2D(15, (1, 1), padding="same", activation="linear",
                  kernel_regularizer=tf.keras.regularizers.l2(l2)),
        name="NIN3")(nin)
    nin = LeakyReLU()(nin)

    nin = kl.TimeDistributed(
        kl.Conv2D(5, (1, 1), padding="same", activation="linear",
                  kernel_regularizer=tf.keras.regularizers.l2(l2)),
        name="NIN4")(nin)
    nin = LeakyReLU()(nin)

    nin = kl.TimeDistributed(
        kl.Conv2D(2, (1, 1), padding="same", activation="linear",
                  kernel_regularizer=tf.keras.regularizers.l2(l2)),
        name="NIN5")(nin)
    nin = LeakyReLU()(nin)

    out = kl.TimeDistributed(
        kl.Conv2D(1, (1, 1), padding="same", activation="sigmoid",
                  kernel_regularizer=tf.keras.regularizers.l2(l2)),
        name="out")(nin)

    # END MODEL #########################################################

    model = tf.keras.Model(inputs=inputs, outputs=out)

    return model


def forecast_clstsm(**kwargs):
    """
    :param kwargs: keyword arguments
        maxpooling_kernel_size: integer. Default 2
            size of max pooling window
        maxpooling_stride: integer. Default 1
            size of max pooling stride
        l2reg: float. Default
            l2 norm used in layers
        input_shape: tuple/list. Length 4. Default

    :return keras.Model
        forecast_lstm model
    """

    # , maxpooling_kernel_size, maxpooling_stride, regularization_rate, resolution
    input_shape = kwargs.get('input_shape', (3, 160, 300, 8))
    maxpooling_stride = kwargs.get('maxpooling_stride', 1)
    regularization_rate = kwargs.get('l2reg', 1e-3)
    maxpooling_kernel_size = kwargs.get('maxpooling_kernel_size', 2)

    resolution = input_shape[1:-1]  # e.g 160, 300

    divisor = maxpooling_stride ** 2
    if resolution[0] % divisor != 0 or resolution[1] % divisor != 0:
        raise Exception("raster resolution values must be a multiple of maxpooling stride x maxpooling stride")

    inputs = tf.keras.Input(shape=input_shape)

    # 1st layer
    x = kl.TimeDistributed(kl.Conv2D(8, (3, 3), padding="same", activation="relu",
                                     kernel_regularizer=tf.keras.regularizers.l2(regularization_rate)),
                           name="seq-conv-1")(inputs)

    c1 = kl.ConvLSTM2D(5, (3, 3), padding="same", activation="relu", name="CLSTM-1", return_sequences=True,
                       kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(x)

    x = kl.TimeDistributed(kl.MaxPooling2D((maxpooling_kernel_size, maxpooling_kernel_size),
                                           (maxpooling_stride, maxpooling_stride)),
                           name="max-pooling-1")(c1)

    # 2nd layer
    x = kl.TimeDistributed(kl.Conv2D(5, (3, 3), padding="same", activation="relu",
                                     kernel_regularizer=tf.keras.regularizers.l2(regularization_rate)), name="seq-conv-2")(x)

    c2 = kl.ConvLSTM2D(3, (3, 3), padding="same", activation="relu", name="CLSTM-2", return_sequences=True,
                       kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(x)

    x = kl.TimeDistributed(kl.MaxPooling2D((maxpooling_kernel_size, maxpooling_kernel_size),
                                           (maxpooling_stride, maxpooling_stride)), name="max-pooling-2")(c2)

    # 3rd layer
    x = kl.TimeDistributed(kl.Conv2D(3, (3, 3), padding="same", activation="relu",
                                     kernel_regularizer=tf.keras.regularizers.l2(regularization_rate)), name="seq-conv-3")(x)

    c3 = kl.ConvLSTM2D(1, (3, 3), padding="same", activation="relu", name="CLSTM-3", return_sequences=True,
                       kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(x)

    # decode layer 1
    d1 = kl.Conv3DTranspose(8, (1, 1, 1), padding="valid", activation="relu", name="up1",
                            kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(c1)

    # decode layer 2
    d2 = kl.Conv3DTranspose(5, (1, 1, 1), padding="valid", activation="relu", name="up2_1",
                            kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(c2)
    d2 = kl.Conv3DTranspose(5, (1, maxpooling_kernel_size, maxpooling_kernel_size),
                            strides=(1, maxpooling_stride, maxpooling_stride),
                            padding="valid", activation="relu", name="up2_2",
                            kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(d2)
    d2 = kl.Conv3DTranspose(8, (1, 1, 1), padding="valid", activation="relu", name="up2_3",
                            kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(d2)

    # decode layer 3
    d3 = kl.Conv3DTranspose(3, (1, 1, 1), padding="valid", activation="relu", name="up3_1",
                            kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(c3)
    d3 = kl.Conv3DTranspose(3, (1, maxpooling_kernel_size, maxpooling_kernel_size),
                            strides=(1, maxpooling_stride, maxpooling_stride),
                            padding="valid", activation="relu", name="up3_2",
                            kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(d3)
    d3 = kl.Conv3DTranspose(5, (1, 1, 1), padding="valid", activation="relu", name="up3_3",
                            kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(d3)
    d3 = kl.Conv3DTranspose(5, (1, maxpooling_kernel_size, maxpooling_kernel_size),
                            strides=(1, maxpooling_stride, maxpooling_stride),
                            padding="valid", activation="relu", name="up3_4",
                            kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(d3)
    d3 = kl.Conv3DTranspose(8, (1, 1, 1), padding="valid", activation="relu", name="up3_5",
                            kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))(d3)

    # concatenate results
    bigdata = kl.concatenate([d1, d2, d3], name="concat_all", axis=-1)


    bigdata = kl.Conv3DTranspose(20, (8, 1, 1), padding="valid",
                                 activation="relu", name="extrapolate_to_30_1",
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                    regularization_rate))(bigdata)

    bigdata = kl.Conv3DTranspose(16, (11, 1, 1),
                                 padding="valid", activation="relu", name="extrapolate_to_30_2",
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                    regularization_rate))(bigdata)

    bigdata = kl.Conv3DTranspose(12, (11, 1, 1),
                                 padding="valid", activation="relu", name="extrapolate_to_30_3",
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                    regularization_rate))(bigdata)

    # channel integration with convolution layers of size 1x1
    nin = kl.TimeDistributed(
        kl.Conv2D(8, (1, 1), padding="same", activation="relu",
                  kernel_regularizer=tf.keras.regularizers.l2(regularization_rate)),
        name="NIN1")(bigdata)

    nin = kl.TimeDistributed(
        kl.Conv2D(4, (1, 1), padding="same", activation="relu",
                  kernel_regularizer=tf.keras.regularizers.l2(regularization_rate)),
        name="NIN2")(nin)

    nin = kl.TimeDistributed(
        kl.Conv2D(2, (1, 1), padding="same", activation="relu",
                  kernel_regularizer=tf.keras.regularizers.l2(regularization_rate)),
        name="NIN3")(nin)

    out = kl.TimeDistributed(
        kl.Conv2D(1, (1, 1), padding="same", activation="sigmoid",
                  kernel_regularizer=tf.keras.regularizers.l2(regularization_rate)),
        name="NIN4")(nin)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    return model


def spatial_feature_pyramid_net_vectorized_ND(**kwargs):
    """
    Creates a model deep CNN for sea ice forecasting, using the Keras functional API.
    Historical input data is encoded using an spatial pyramid network
        https://arxiv.org/pdf/1606.00915.pdf
        https://arxiv.org/pdf/1612.03144.pdf
        https://arxiv.org/pdf/1612.01105.pdf
    Once each input data has been independently encoded though the spatial feature pyramid,
        the historical input data is further encoded into a single feature-cube using a convolutional-LSTM with
        return_sequence=False.
        https://arxiv.org/pdf/1506.04214.pdf
        This tensor is also concatenated with the latest day of historical input data to
        preserve certain input features for which only the most recent day is important.
    From the merged encoded state, a sequence of output steps is produced using the custom ResStepDecoder layer
        ResStepDecoder may be described as:
            ResStepDecoder(inputEncodedState=E):
                S[0] = G(E)
                for i in (1, Sequence_Length) do:
                    S[i] := S[i-1] + F(E, S[i-1])

                return S[1:]

            Where G is a learned function to estimate an initial state from the encoded state E.
            Where F is s learned function to predict the delta between each subsequent time-step.

        ResStepDecoder makes use  of TensorFlow's highly optimized SeparableConv2D to increase computation efficiency
        and reduce the total number of parameters.

    Finally, the sequence of extrapolated states are converted into the ice-presence probability through a
        time-distributed network-in-a-network structure, where the final layer uses a Sigmoid activation function.

    For use with ensembles.

    Matthew King, November 2019 @ NRC
    :return: Keras model object
    """
    if 'l2reg' in kwargs:
        l2 = kwargs['l2reg']
    else:
        l2 = 0.001

    if 'input_shape' in kwargs:
        input_shape = kwargs['input_shape']
    else:
        input_shape = (3, 160, 300, 8)

    if 'output_steps' in kwargs:
        output_steps = kwargs['output_steps']
        if type(output_steps) != int:
            raise TypeError('Received output_steps of non-int type')
    else:
        output_steps = 30

    if 'leaky_relu_alpha' in kwargs:
        alpha = kwargs['leaky_relu_alpha']
        if type(alpha) != float:
            raise TypeError('Received leaky relu alpha of non-float type')
    else:
        alpha = 0.01

    if 'debug' in kwargs:
        debug = kwargs['debug']
        if type(debug) != bool:
            raise TypeError('Received debug of non-bool type')
    else:
        debug = False

    inputs = tf.keras.Input(shape=input_shape)

    n_features = 24

    full_res_map = mu.spatial_feature_pyramid(inputs, n_features, (3, 3), 8, alpha=alpha, l2_rate=l2,
                                              return_all=False, debug=debug)

    encoded_state = kl.ConvLSTM2D(48-input_shape[-1], (3, 3), padding='same', activation='selu',
                              kernel_initializer='lecun_normal', kernel_regularizer=tf.keras.regularizers.l2(l2),
                              name='State_Encoder')(full_res_map)

    days_in = input_shape[0]
    input_last_day_only = kl.Cropping3D(cropping=((days_in-1, 0), (0, 0), (0, 0)))(inputs)
    input_last_day_only = kl.Reshape(target_shape=list(input_shape[1:]))(input_last_day_only) #remove time axis
    encoded_state = kl.concatenate([encoded_state, input_last_day_only], axis=-1)

    if debug:
        print('ENCODED STATE')
        print(encoded_state.shape.as_list())

    # x = mu.ResStepDecoder(16, 48, 60, (3,3), 2, l2, alpha, name='MyResStepDecoder')(encoded_state)
    x = mu.res_step_decoder_functional(encoded_state, filters=16, upsampled_filters=48, output_steps=output_steps,
                                       kernel_size=(3, 3), depth_multiplier=2, l2_rate=l2, alpha=alpha,
                                       return_sequence=True, anchored=True)

    x = kl.TimeDistributed(kl.Conv2D(48, (1, 1), activation='linear', padding='same', name='nin1',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))(x)
    x = LeakyReLU(alpha)(x)
    x = kl.TimeDistributed(kl.Conv2D(32, (1, 1), activation='linear', padding='same', name='nin2',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))(x)
    x = LeakyReLU(alpha)(x)
    x = kl.TimeDistributed(kl.Conv2D(16, (1, 1), activation='linear', padding='same', name='nin3',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))(x)
    x = LeakyReLU(alpha)(x)

    x = kl.TimeDistributed(kl.Conv2D(8, (1,1), activation='sigmoid', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)),
                           name='sigmoid_pre_out')(x)

    x = kl.TimeDistributed(kl.Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)),
                           name='sigmoid_out')(x)
    out = x

    return tf.keras.Model(inputs=inputs, outputs=out)


def spatial_feature_pyramid_net_hiddenstate_ND(**kwargs):
    """
    Creates a model deep CNN for sea ice forecasting, using the Keras functional API.
    Historical input data is encoded using an spatial pyramid network
        https://arxiv.org/pdf/1606.00915.pdf
        https://arxiv.org/pdf/1612.03144.pdf
        https://arxiv.org/pdf/1612.01105.pdf
    Once each input data has been independently encoded though the spatial feature pyramid,
        the historical input data is further encoded into a single feature-cube using a convolutional-LSTM with
        return_sequence=False.
        https://arxiv.org/pdf/1506.04214.pdf
        This tensor is also concatenated with the latest day of historical input data to
        preserve certain input features for which only the most recent day is important.
    The primary difference between this model and spatial_feature_pyramid_net_vectorized_ND is the inclusion
        of a hidden state during the recurrent decoder stage.
        The advantage of this hidden state is to represent factors which may change over time and drive/be correlated
        to the desired process without being directly related to the output.
    From the merged encoded state, a sequence of output steps is produced using the custom ResStepDecoderHS layer
        ResStepDecoder may be described as:
            ResStepDecoderHS(inputEncodedState=E):
                S[0] = G(E)
                HS = Q(E)
                for i in (1, Sequence_Length) do:
                    S[i] := S[i-1] + F(E, S[i-1], HS)
                    HS := HS + V(E, S[i-1], HS)

                return S[1:]

            Where G is a learned function to estimate an initial state from the encoded state E.
            Where F is s learned function to predict the delta between each subsequent time-step.

        ResStepDecoder makes use  of TensorFlow's highly optimized SeparableConv2D to increase computation efficiency
        and reduce the total number of parameters.

    Finally, the sequence of extrapolated states are converted into the ice-presence probability through a
        time-distributed network-in-a-network structure, where the final layer uses a Sigmoid activation function.

    Moderate candidate and good for use with ensembles.
    Best of the non-forecast channel augmented models.

    Matthew King, November 2019 @ NRC
    :return: Keras model object
    """
    if 'l2reg' in kwargs:
        l2 = kwargs['l2reg']
    else:
        l2 = 0.001

    if 'input_shape' in kwargs:
        input_shape = kwargs['input_shape']
    else:
        input_shape = (3, 160, 300, 8)

    if 'output_steps' in kwargs:
        output_steps = kwargs['output_steps']
        if type(output_steps) != int:
            raise TypeError('Received output_steps of non-int type')
    else:
        output_steps = 30

    if 'leaky_relu_alpha' in kwargs:
        alpha = kwargs['leaky_relu_alpha']
        if type(alpha) != float:
            raise TypeError('Received leaky relu alpha of non-float type')
    else:
        alpha = 0.01
    if 'debug' in kwargs:
        debug = kwargs['debug']
        if type(debug) != bool:
            raise TypeError('Received debug of non-bool type')
    else:
        debug = False

    inputs = tf.keras.Input(shape=input_shape)

    n_features = 24

    full_res_map = mu.spatial_feature_pyramid(inputs, n_features, (3, 3), 8, alpha=alpha, l2_rate=l2,
                                              return_all=False, debug=debug)

    encoded_state = kl.ConvLSTM2D(48-input_shape[-1], (3, 3), padding='same', activation='selu',
                              kernel_initializer='lecun_normal', kernel_regularizer=tf.keras.regularizers.l2(l2),
                              name='State_Encoder')(full_res_map)

    days_in = input_shape[0]
    input_last_day_only = kl.Cropping3D(cropping=((days_in-1, 0), (0, 0), (0, 0)))(inputs)
    input_last_day_only = kl.Reshape(target_shape=list(input_shape[1:]))(input_last_day_only) #remove time axis
    encoded_state = kl.concatenate([encoded_state, input_last_day_only], axis=-1)

    if debug:
        print('ENCODED STATE')
        print(encoded_state.shape.as_list())

    # x = mu.ResStepDecoder(16, 48, 60, (3,3), 2, l2, alpha, name='MyResStepDecoder')(encoded_state)
    x = mu.res_step_decoder_HS_functional(encoded_state, filters=16, hidden_filters=16, upsampled_filters=48,
                                          output_steps=output_steps,
                                       kernel_size=(3, 3), depth_multiplier=2, l2_rate=l2, alpha=alpha,
                                       return_sequence=True, anchored=True)

    x = kl.TimeDistributed(kl.Conv2D(48, (1, 1), activation='linear', padding='same', name='nin1',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))(x)
    x = LeakyReLU(alpha)(x)
    x = kl.TimeDistributed(kl.Conv2D(32, (1, 1), activation='linear', padding='same', name='nin2',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))(x)
    x = LeakyReLU(alpha)(x)
    x = kl.TimeDistributed(kl.Conv2D(16, (1, 1), activation='linear', padding='same', name='nin3',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))(x)
    x = LeakyReLU(alpha)(x)

    x = kl.TimeDistributed(kl.Conv2D(8, (1,1), activation='sigmoid', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)),
                           name='sigmoid_pre_out')(x)

    x = kl.TimeDistributed(kl.Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)),
                           name='sigmoid_out')(x)
    out = x

    return tf.keras.Model(inputs=inputs, outputs=out)

def spatial_feature_pyramid_anomaly(**kwargs):
    """
    Similar to spatial_feature_pyramid_net_hiddenstate_ND but it also takes a second input (from climate normal) and subtract it from the first input to get anomaly features. The anomaly is appended to the first input to make the combined input. 
    
    """
    
    if 'l2reg' in kwargs:
        l2 = kwargs['l2reg']
    else:
        l2 = 0.001

    if 'input_shape' in kwargs:
        input_shape = kwargs['input_shape']
    else:
        input_shape = (3, 160, 300, 8)
    
    if 'anomaly_shape' in kwargs:
        anomaly_shape = kwargs['anomaly_shape']
    else:
        anomaly_shape = (3, 160, 300, 8)

    if 'output_steps' in kwargs:
        output_steps = kwargs['output_steps']
        if type(output_steps) != int:
            raise TypeError('Received output_steps of non-int type')
    else:
        output_steps = 30

    if 'leaky_relu_alpha' in kwargs:
        alpha = kwargs['leaky_relu_alpha']
        if type(alpha) != float:
            raise TypeError('Received leaky relu alpha of non-float type')
    else:
        alpha = 0.01
    if 'debug' in kwargs:
        debug = kwargs['debug']
        if type(debug) != bool:
            raise TypeError('Received debug of non-bool type')
    else:
        debug = False

    inputs = tf.keras.Input(shape=input_shape)
    second_input = tf.keras.Input(shape=anomaly_shape)
    
    anomaly_input = kl.Subtract()([inputs,second_input])
    combined_inputs = kl.concatenate([inputs,anomaly_input], axis=-1)
    

    n_features = 24

    full_res_map = mu.spatial_feature_pyramid(combined_inputs, n_features, (3, 3), 8, alpha=alpha, l2_rate=l2,
                                              return_all=False, debug=debug)

    encoded_state = kl.ConvLSTM2D(48-input_shape[-1], (3, 3), padding='same', activation='selu',
                              kernel_initializer='lecun_normal', kernel_regularizer=tf.keras.regularizers.l2(l2),
                              name='State_Encoder')(full_res_map)

    days_in = input_shape[0]
    input_last_day_only = kl.Cropping3D(cropping=((days_in-1, 0), (0, 0), (0, 0)))(combined_inputs)
    input_last_day_only = kl.Reshape(target_shape=list(combined_inputs.shape[2:]))(input_last_day_only) #remove time axis
    encoded_state = kl.concatenate([encoded_state, input_last_day_only], axis=-1)

    if debug:
        print('ENCODED STATE')
        print(encoded_state.shape.as_list())

    # x = mu.ResStepDecoder(16, 48, 60, (3,3), 2, l2, alpha, name='MyResStepDecoder')(encoded_state)
    x = mu.res_step_decoder_HS_functional(encoded_state, filters=16, hidden_filters=16, upsampled_filters=48,
                                          output_steps=output_steps,
                                       kernel_size=(3, 3), depth_multiplier=2, l2_rate=l2, alpha=alpha,
                                       return_sequence=True, anchored=True)

    x = kl.TimeDistributed(kl.Conv2D(48, (1, 1), activation='linear', padding='same', name='nin1',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))(x)
    x = LeakyReLU(alpha)(x)
    x = kl.TimeDistributed(kl.Conv2D(32, (1, 1), activation='linear', padding='same', name='nin2',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))(x)
    x = LeakyReLU(alpha)(x)
    x = kl.TimeDistributed(kl.Conv2D(16, (1, 1), activation='linear', padding='same', name='nin3',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)))(x)
    x = LeakyReLU(alpha)(x)

    x = kl.TimeDistributed(kl.Conv2D(8, (1,1), activation='sigmoid', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)),
                           name='sigmoid_pre_out')(x)

    x = kl.TimeDistributed(kl.Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2)),
                           name='sigmoid_out')(x)
    out = x

    return tf.keras.Model(inputs=[inputs,second_input], outputs=out)