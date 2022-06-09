"""
Support for models

"""

import math
from tensorflow import keras
import tensorflow.keras.layers as kl
from tensorflow.keras.layers import LeakyReLU


def spatial_feature_pyramid(input_sequence, full_res_features, kernel_size, max_downsampling_factor, alpha, l2_rate,
                            return_all=False, name_extension="", **kwargs):
    """
    Feature extractor network. Applied independently on a sequence of inputs.


    :param input_sequence: Input tensor. 5D tensor [batch_size, time-steps, Height, Width, Channels]
    :param full_res_features: The number of features to be extracted at full resolution.
                            Each downsampled feature map will have int(base_features/downsampling_factor) features
    :param kernel_size: tuple, kernel size for using in convolutional layers
                        e.g (3,3)
    :param max_downsampling_factor: int. Maximum downsampling factor to be applied. Must be a power of 2 and > 1.
                        e.g 8
    :param alpha: float. Alpha value for use with Leaky ReLU
    :param l2_rate: float. l2 weight regularization rate
    :param return_all: boolean, optional. Default False.
                            True if all resolution feature maps should be returned, otherwise only full resolution.
    :param name_extension: string. Added to each layer's name
    :param kwargs: keyword arguments
                debug: boolean
    :return: Tensor or List of Tensors
    """

    assert(len(input_sequence.shape.as_list()) == 5), 'input_sequence must be a 5D tensor'
    assert(type(full_res_features) == int), 'base_features must be an int'
    assert(type(kernel_size) == tuple and len(kernel_size) == 2), 'kernel_size must be a tuple with len 2'
    assert(type(kernel_size[0]) == int and type(kernel_size[1]) == int), 'Each value of kernel_size must be an int'
    assert(type(alpha) == float), 'alhpa must be a float'
    assert(type(l2_rate) == float), 'l2_rate must be a float'
    assert(type(return_all) == bool), 'return_all must be a bool'
    assert(type(max_downsampling_factor) == int and max_downsampling_factor > 1), 'max_downsampling_factor must be \'' \
                                                                                  'an int greater than 1'
    assert(math.log2(max_downsampling_factor).is_integer()), 'max_downsampling_factor must be a power of 2'
    assert(max_downsampling_factor <= full_res_features), 'max_downsampling_factor must not be greater than base_features'

    if 'debug' in kwargs:
        debug = kwargs['debug']
        assert(type(debug) == bool), 'debug must be a bool'
    else:
        debug = False

    base_features = kl.TimeDistributed(
        kl.Conv2D(full_res_features, kernel_size, padding='same', kernel_regularizer=keras.regularizers.l2(l2_rate),
                  kernel_initializer='orthogonal'),
        name='Conv2D_FullSize'+name_extension)(input_sequence)
    base_features = LeakyReLU(alpha)(base_features)

    feature_maps = [base_features]

    exp2 = int(math.log2(max_downsampling_factor))
    downsampling_factors = [2 ** x for x in range(1, exp2+1)]  # [2, 4, 8, ... max_downsampling_factor]

    for downsampling_factor in downsampling_factors:
        # produce a feature map at each downsampling size
        downsampled = kl.TimeDistributed(
            kl.AveragePooling2D((downsampling_factor, downsampling_factor),
                                strides=(downsampling_factor, downsampling_factor),
                                padding='same'),  # padding=same ensures exactly half, quarter, etc. Not actually same.
            name="Downsampling2D_{}".format(downsampling_factor)+name_extension
        )(input_sequence)
        downsampled = kl.TimeDistributed(
            kl.Conv2D(int(full_res_features / downsampling_factor), kernel_size, padding='same',
                      kernel_regularizer=keras.regularizers.l2(l2_rate)),
            name='Conv2d_Downsampled_{}'.format(downsampling_factor)+name_extension
        )(downsampled)
        downsampled = LeakyReLU(alpha)(downsampled)
        feature_maps.append(downsampled)

    if debug:
        print('FEATURE MAPS')
        print([f.shape.as_list() for f in feature_maps])

    updated_feature_maps = [feature_maps[-1]]
    for level in range(len(feature_maps) - 1, 0, -1):  # 3,2,1
        # use the lower resolution feature map to update the feature map at the higher resolution
        f = feature_maps[level]
        n = feature_maps[level - 1]
        features_at_n = int(full_res_features / (2 ** (level - 1)))

        fp = kl.TimeDistributed(
            kl.Conv2DTranspose(features_at_n, (4, 4), (2, 2), padding='same',
                               kernel_regularizer=keras.regularizers.l2(l2_rate)),
            name='Pyramid_upflow_{}'.format(level)+name_extension)(f)
        fp = LeakyReLU(alpha)(fp)
        fp_shape = fp.shape.as_list()
        n_shape = n.shape.as_list()
        if debug:
            print("fp_type {}".format(type(fp)))
            print("fp_shape {}".format(fp_shape))
            print("n_shape {}".format(n_shape))
        if not fp_shape == n_shape:
            dh = fp_shape[-2] - n_shape[-2]
            dw = fp_shape[-3] - n_shape[-3]
            fp = kl.Cropping3D(cropping=((0, 0), (0, dw), (0, dh)),
                               name='Upflow_Cropping_{}'.format(level)+name_extension)(fp)
        n = kl.Add()([n, fp])
        feature_maps[level - 1] = n
        updated_feature_maps.append(n)

    if debug:
        print('Updated FEATURE MAPS')
        print([f.shape.as_list() for f in updated_feature_maps])

    if return_all:
        return updated_feature_maps
    else:
        return updated_feature_maps[-1]  # full res only


def spatial_feature_pyramid_separable(input_sequence, full_res_features, kernel_size, max_downsampling_factor,
                                      alpha, l2_rate, depth_multiplier=4,
                                      return_all=False, name_extension="", **kwargs):
    """
    Feature extractor network. Applied independently on a sequence of inputs. Uses Separable Conv2D variant for higher
    computational efficiency.

    :param input_sequence: Input tensor. 5D tensor [batch_size, time-steps, Height, Width, Channels]
    :param full_res_features: The number of features to be extracted at full resolution.
                            Each downsampled feature map will have int(base_features/downsampling_factor) features
    :param kernel_size: tuple, kernel size for using in convolutional layers
                        e.g (3,3)
    :param max_downsampling_factor: int. Maximum downsampling factor to be applied. Must be a power of 2 and > 1.
                        e.g 8
    :param alpha: float. Alpha value for use with Leaky ReLU
    :param l2_rate: float. l2 weight regularization rate
    :param return_all: boolean, optional. Default False.
                            True if all resolution feature maps should be returned, otherwise only full resolution.
    :param name_extension: string. Added to each layer's name
    :param kwargs: keyword arguments
                debug: boolean
    :return: Tensor or List of Tensors
    """

    assert(len(input_sequence.shape.as_list()) == 5), 'input_sequence must be a 5D tensor'
    assert(type(full_res_features) == int), 'base_features must be an int'
    assert(type(kernel_size) == tuple and len(kernel_size) == 2), 'kernel_size must be a tuple with len 2'
    assert(type(kernel_size[0]) == int and type(kernel_size[1]) == int), 'Each value of kernel_size must be an int'
    assert(type(alpha) == float), 'alhpa must be a float'
    assert(type(l2_rate) == float), 'l2_rate must be a float'
    assert(type(return_all) == bool), 'return_all must be a bool'
    assert(type(max_downsampling_factor) == int and max_downsampling_factor > 1), 'max_downsampling_factor must be \'' \
                                                                                  'an int greater than 1'
    assert(math.log2(max_downsampling_factor).is_integer()), 'max_downsampling_factor must be a power of 2'
    assert(max_downsampling_factor <= full_res_features), 'max_downsampling_factor must not be greater than base_features'

    if 'debug' in kwargs:
        debug = kwargs['debug']
        assert(type(debug) == bool), 'debug must be a bool'
    else:
        debug = False

    base_features = kl.TimeDistributed(
        kl.SeparableConv2D(full_res_features, kernel_size, depth_multiplier=depth_multiplier,
                           padding='same', kernel_regularizer=keras.regularizers.l2(l2_rate),
                           depthwise_regularizer=keras.regularizers.l2(l2_rate),
                           kernel_initializer='orthogonal'),
        name='SepConv2D_FullSize'+name_extension)(input_sequence)
    base_features = LeakyReLU(alpha)(base_features)

    feature_maps = [base_features]

    exp2 = int(math.log2(max_downsampling_factor))
    downsampling_factors = [2 ** x for x in range(1, exp2+1)]  # [2, 4, 8, ... max_downsampling_factor]

    for downsampling_factor in downsampling_factors:
        # produce a feature map at each downsampling size
        downsampled = kl.TimeDistributed(
            kl.AveragePooling2D((downsampling_factor, downsampling_factor),
                                strides=(downsampling_factor, downsampling_factor),
                                padding='same'),  # padding=same ensures exactly half, quarter, etc. Not actually same.
            name="Downsampling2D_{}".format(downsampling_factor)+name_extension
        )(input_sequence)
        downsampled = kl.TimeDistributed(
            kl.SeparableConv2D(int(full_res_features / downsampling_factor), kernel_size,
                               depth_multiplier=depth_multiplier,
                               padding='same', depthwise_regularizer=keras.regularizers.l2(l2_rate),
                               kernel_regularizer=keras.regularizers.l2(l2_rate)),
            name='SepConv2d_Downsampled_{}'.format(downsampling_factor)+name_extension
        )(downsampled)
        downsampled = LeakyReLU(alpha)(downsampled)
        feature_maps.append(downsampled)

    if debug:
        print('FEATURE MAPS')
        print([f.shape.as_list() for f in feature_maps])

    updated_feature_maps = [feature_maps[-1]]
    for level in range(len(feature_maps) - 1, 0, -1):  # 3,2,1
        # use the lower resolution feature map to update the feature map at the higher resolution
        f = feature_maps[level]
        n = feature_maps[level - 1]
        features_at_n = int(full_res_features / (2 ** (level - 1)))

        fp = kl.TimeDistributed(
            kl.Conv2DTranspose(features_at_n, (4, 4), (2, 2), padding='same',
                               kernel_regularizer=keras.regularizers.l2(l2_rate)),
            name='Pyramid_upflow_{}'.format(level)+name_extension)(f)
        fp = LeakyReLU(alpha)(fp)
        fp_shape = fp.shape.as_list()
        n_shape = n.shape.as_list()
        if debug:
            print("fp_type {}".format(type(fp)))
            print("fp_shape {}".format(fp_shape))
            print("n_shape {}".format(n_shape))
        if not fp_shape == n_shape:
            dh = fp_shape[-2] - n_shape[-2]
            dw = fp_shape[-3] - n_shape[-3]
            fp = kl.Cropping3D(cropping=((0, 0), (0, dw), (0, dh)),
                               name='Upflow_Cropping_{}'.format(level)+name_extension)(fp)
        n = kl.Add()([n, fp])
        feature_maps[level - 1] = n
        updated_feature_maps.append(n)

    if debug:
        print('Updated FEATURE MAPS')
        print([f.shape.as_list() for f in updated_feature_maps])

    if return_all:
        return updated_feature_maps
    else:
        return updated_feature_maps[-1]  # full res only


def res_step_decoder_functional(input_encoded_state, filters, upsampled_filters, output_steps, kernel_size=(3, 3),
                                depth_multiplier=2, l2_rate=1e-4, alpha=3e-2, return_sequence=True, anchored=True,
                                **kwargs):
    """
    A 'layer' which extrapolates a given state across the given number of steps.

    :param input_encoded_state: Input tensor, of shape [batch_size, height, width, channels]
    :param filters: The number of output filters/data channels
    :param upsampled_filters: The number of channels to be used during the update step
    :param output_steps:  The number of timesteps to be processed
    :param kernel_size: kernel size e.g (3,3)s
    :param depth_multiplier: Depth multiplier for SeperableConv layers
    :param l2_rate: l2 weight regularization rate
    :param alpha: alpha for LeakyRelu
    :param return_sequence: True if the whole sequence should be returned, False for only the final state
    :param anchored: True if the update step is computed within context of the input Encoded state, \
                        False for independent update steps
    :param kwargs: keyword arguments
                debug: boolean
    :return:Output tensor
    """

    if 'debug' in kwargs:
        debug = kwargs['debug']
    else:
        debug = False

    assert len(input_encoded_state.shape.as_list()) == 4

    initial_state = kl.SeparableConv2D(filters, kernel_size, padding='same', depth_multiplier=depth_multiplier,
                                       kernel_regularizer=keras.regularizers.l2(l2_rate),
                                       pointwise_regularizer=keras.regularizers.l2(l2_rate))(input_encoded_state)
    initial_state = kl.LeakyReLU(alpha)(initial_state)

    daily_extrapolated_states = []  # placeholder

    upsampler = kl.Conv2D(upsampled_filters, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(l2_rate),
                          name='Incoming_State_Upsampler')  # NIN

    res_pred = kl.SeparableConv2D(filters, kernel_size, padding='same', kernel_regularizer=keras.regularizers.l2(l2_rate),
                                  pointwise_regularizer=keras.regularizers.l2(l2_rate),
                                  name='Residual_Predictor', depth_multiplier=depth_multiplier)

    inner_concatenator = kl.Concatenate(axis=-1, name='inner_concatenator')  # stacks along the feature dimension

    adder = kl.Add(name='residual_adder')

    expand_dims = kl.Lambda(lambda in_tensor: keras.backend.expand_dims(in_tensor, axis=1))

    leakyRelu = kl.LeakyReLU(alpha, name='Leaky_ReLU')

    for i in range(output_steps):
        if i == 0:
            incoming_state = initial_state
        else:
            incoming_state = daily_extrapolated_states[-1]

        upsampled = upsampler(incoming_state)
        upsampled = leakyRelu(upsampled)

        if anchored:
            combined_state = inner_concatenator([upsampled, input_encoded_state])
            residual = res_pred(combined_state)
        else:
            residual = res_pred(upsampled)

        next_state = adder([incoming_state, residual])
        daily_extrapolated_states.append(next_state)

    if debug:
        print('Daily States')
        print(daily_extrapolated_states)

    if return_sequence:
        # Expand dims to add time-step dimension
        for i in range(len(daily_extrapolated_states)):
            s = daily_extrapolated_states[i]
            s = expand_dims(s)
            daily_extrapolated_states[i] = s

        return kl.Concatenate(axis=1)(daily_extrapolated_states)  # concatenate along the time-step dimension
    else:
        return daily_extrapolated_states[-1]

def res_step_decoder_HS_functional(input_encoded_state, filters, hidden_filters, upsampled_filters, output_steps,
                                   kernel_size=(3, 3), depth_multiplier=2, l2_rate=1e-4, alpha=3e-2,
                                   return_sequence=True, anchored=True, **kwargs):
    """
    A 'layer' which extrapolates a given state across the given number of steps. Uses a hidden state for greater
    internal representation power.

    :param input_encoded_state: Input tensor, of shape [batch_size, height, width, channels]
    :param filters: The number of output filters/data channels
    :param hidden_filters: The number of channels in the hidden state
    :param upsampled_filters: The number of channels to be used during the update step
    :param output_steps:  The number of timesteps to be processed
    :param kernel_size: kernel size e.g (3,3)s
    :param depth_multiplier: Depth multiplier for SeperableConv layers
    :param l2_rate: l2 weight regularization rate
    :param alpha: alpha for LeakyRelu
    :param return_sequence: True if the whole sequence should be returned, False for only the final state
    :param anchored: True if the update step is computed within context of the input Encoded state, \
                        False for independent update steps
    :param kwargs: keyword arguments
                debug: boolean
    :return:Output tensor
    """

    if 'debug' in kwargs:
        debug = kwargs['debug']
    else:
        debug = False

    assert len(input_encoded_state.shape.as_list()) == 4

    initial_state = kl.SeparableConv2D(filters, kernel_size, padding='same', depth_multiplier=depth_multiplier,
                                       kernel_regularizer=keras.regularizers.l2(l2_rate),
                                       pointwise_regularizer=keras.regularizers.l2(l2_rate),
                                       name='Output_State_Initializer')(input_encoded_state)
    initial_state = kl.LeakyReLU(alpha)(initial_state)

    hidden_state = kl.SeparableConv2D(hidden_filters, kernel_size, padding='same', depth_multiplier=depth_multiplier,
                                       kernel_regularizer=keras.regularizers.l2(l2_rate),
                                       pointwise_regularizer=keras.regularizers.l2(l2_rate),
                                       name='Hidden_State_Initializer')(input_encoded_state)
    hidden_state = kl.LeakyReLU(alpha)(hidden_state)

    daily_extrapolated_states = []  # placeholder

    upsampler = kl.Conv2D(upsampled_filters, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(l2_rate),
                          name='Incoming_State_Upsampler')  # NIN
    hidden_upsampler = kl.Conv2D(upsampled_filters, (1, 1), padding='same',
                                 kernel_regularizer=keras.regularizers.l2(l2_rate),
                                 name='Hidden_State_Upsampler')

    res_pred = kl.SeparableConv2D(filters, kernel_size, padding='same', kernel_regularizer=keras.regularizers.l2(l2_rate),
                                  pointwise_regularizer=keras.regularizers.l2(l2_rate),
                                  name='Residual_Predictor', depth_multiplier=depth_multiplier)
    hidden_res_pred = kl.SeparableConv2D(hidden_filters, kernel_size, padding='same',
                                         kernel_regularizer=keras.regularizers.l2(l2_rate),
                                         pointwise_regularizer=keras.regularizers.l2(l2_rate),
                                         name='Hidden_Residual_Predictor', depth_multiplier=depth_multiplier)

    inner_concatenator = kl.Concatenate(axis=-1, name='inner_concatenator')  # stacks along the feature dimension

    adder1 = kl.Add(name='residual_adder')
    adder2 = kl.Add(name='hidden_adder')

    expand_dims = kl.Lambda(lambda in_tensor: keras.backend.expand_dims(in_tensor, axis=1))

    leakyRelu = kl.LeakyReLU(alpha, name='Leaky_ReLU')

    for i in range(output_steps):
        if i == 0:
            incoming_state = initial_state
        else:
            incoming_state = daily_extrapolated_states[-1]

        upsampled = upsampler(incoming_state)
        upsampled = leakyRelu(upsampled)
        hidden_upsampled = hidden_upsampler(hidden_state)

        if anchored:
            combined_state = inner_concatenator([upsampled, input_encoded_state, hidden_upsampled])
            residual = res_pred(combined_state)
            hidden_residual = hidden_res_pred(combined_state)
        else:
            residual = res_pred(upsampled, hidden_upsampled)
            hidden_residual = hidden_res_pred(upsampled, hidden_upsampled)

        next_state = adder1([incoming_state, residual])
        hidden_state = adder2([hidden_state, hidden_residual])

        daily_extrapolated_states.append(next_state)

    if debug:
        print('Daily States')
        print(daily_extrapolated_states)

    if return_sequence:
        # Expand dims to add time-step dimension
        for i in range(len(daily_extrapolated_states)):
            s = daily_extrapolated_states[i]
            s = expand_dims(s)
            daily_extrapolated_states[i] = s

        # concatenate along the time-step dimension
        return kl.Concatenate(axis=1, name='Output_Sequence')(daily_extrapolated_states)
    else:
        return daily_extrapolated_states[-1]


class ResStepDecoder(kl.Layer):
    """
    ResStepDecoder
    A custom rnn-like layer which learns a function to extrapolate a given state forward across N time steps.
    To the best of my knowledge, a novel layer architecture.
    Matthew King, November 2019.

    Even though this implementation is per Keras API specifications, it runs much slower than the messier alternative.
    For now, use res_step_decoder_functional() instead
    """

    def __init__(self, filters, upsampled_filters, output_steps, kernel_size=(3, 3), depth_multiplier=2, l2_rate=1e-4,
                 alpha=3e-2, return_sequence=True, anchored=True, **kwargs):
        """
        :param filters: The number of output filters/data channels
        :param upsampled_filters: The number of channels to be used during the update step
        :param output_steps:  The number of timesteps to be processed
        :param kernel_size: kernel size e.g (3,3)s
        :param depth_multiplier: Depth multiplier for SeperableConv layers
        :param l2_rate: l2 weight regularization rate
        :param alpha: alpha for LeakyRelu
        :param return_sequence: True if the whole sequence should be returned, False for only the final state
        :param anchored: True if the update step is computed within context of the input Encoded state, \
                            False for independent update steps
        :param kwargs: keyword arguments
        """
        super(ResStepDecoder, self).__init__(**kwargs)
        assert(type(filters) == int), 'filters must be an int'
        assert(type(upsampled_filters) == int), 'upsampled_filters must be an int'
        assert(type(output_steps) == int and output_steps > 0), 'output steps must be a positive integer'
        # assert(type(kernel_size) == tuple and len(kernel_size) == 2), 'kernel_size must be tuple of length 2'
        # assert(type(kernel_size[0]) == int and kernel_size[0] > 0), 'kernel_size[0] must be positive integer'
        # assert(type(kernel_size[1]) == int and kernel_size[1] > 0), 'kernel_size'
        self.filters = filters
        self.upsampled_filters = upsampled_filters
        self.output_steps = output_steps
        self.kernel_size = kernel_size
        self.depth_multiplier = depth_multiplier
        self.l2_rate = l2_rate
        self.alpha = alpha
        self.return_sequence = return_sequence
        self.anchored = anchored

        self.initial_state_estimator = kl.SeparableConv2D(filters, kernel_size=kernel_size,
                                                          depth_multiplier=depth_multiplier, padding='same',
                                                          activity_regularizer=keras.regularizers.l2(l2_rate))

        self.upSampler = kl.Conv2D(upsampled_filters, (1, 1), padding='same',
                                   kernel_regularizer=keras.regularizers.l2(l2_rate))

        self.res_pred = kl.SeparableConv2D(filters, kernel_size=kernel_size, depth_multiplier=depth_multiplier,
                                           kernel_regularizer=keras.regularizers.l2(l2_rate), padding='same')
        self.adder = kl.Add()
        self.inner_concatenator = kl.Concatenate(axis=-1)  # feature dimension, assuming channels last
        self.outer_concatenator = kl.Concatenate(axis=1)  # time step dimension
        self.expand_dims = kl.Lambda(lambda in_tensor: keras.backend.expand_dims(in_tensor, axis=1))
        self.leaky = kl.LeakyReLU(alpha)

    def get_config(self):
        base_config = super(ResStepDecoder, self).get_config()
        base_config['filters'] = self.filters
        base_config['output_steps'] = self.output_steps
        base_config['upsampled_filters'] = self.upsampled_filters
        base_config['kernel_size'] = self.kernel_size
        base_config['depth_multiplier'] = self.depth_multiplier
        base_config['l2_rate'] = self.l2_rate
        base_config['alpha'] = self.alpha
        base_config['return_sequence'] = self.return_sequence
        base_config['anchored'] = self.anchored
        return base_config

    def call(self, inputs, **kwargs):
        """
        :param inputs: input tensor. A 4D tensor with shape (batch_size, Height, Width, Channels)
                            The encoded state to be extrapolated across N time steps.
        :param kwargs: kwargs
        :return: output tensor.
                If return_sequence=True, a 5D tensor with shape (batch_size, output_steps, Height, Width, filters)
                else, a 4D tensor with shape (batch_size, Height, Width, filters)
        """

        # TODO: validate input tensor shape
        """
        S[0] = G(E)
        """
        initial_state = self.initial_state_estimator(inputs)
        states = []
        for i in range(self.output_steps):
            """
            if anchored:
                S[k] = S[k-1] + F(E, S[k-1])
            else:
                S[k] = S[k-1] + F(S[k-1])
            """
            if i == 0:
                incoming_state = initial_state

            else:
                incoming_state = states[-1]

            upsampled = self.upSampler(incoming_state)
            upsampled = self.leaky(upsampled)
            if self.anchored:
                concatenated = self.inner_concatenator([upsampled, inputs])
                residual = self.res_pred(concatenated)
            else:
                residual = self.res_pred(upsampled)
            next_state = self.adder([incoming_state, residual])
            states.append(next_state)

        if self.return_sequence:
            for i in range(self.output_steps):
                s = states[i]
                s = self.expand_dims(s)
                states[i] = s

            return self.outer_concatenator(states)
        else:
            return states[-1]


# this is necessary for loading custom layers with keras.models.load_model()
# only if that model actually uses the layer will this be used, but otherwise it is harmless.
# also fixed a bug NameError: name 'keras' is not defined
custom_objects = {'ResStepDecoder': ResStepDecoder, 'keras': keras}

# def GAN_Encoder(c_dim,growthRate,depth,bottleneck,reduction,gf_dim):
#     nDenseBlocks = (depth-4) // 3  #(22-4)//3=6
#     if bottleneck:
#         nDenseBlocks = nDenseBlocks
#     #define input and output channel
#     nOut_conv1 = growthRate
#     nIn_dense1 = nOut_conv1
#     nOut_dense1 = nOut_conv1 + nDenseBlocks * growthRate
#     nIn_trans1 = nOut_dense1
#     nOut_trans1 = int(math.floor(nOut_dense1 * reduction))
#     nIn_dense2 = nOut_trans1
#     nOut_dense2 = nOut_trans1 + nDenseBlocks * growthRate
#     nIn_trans2 = nOut_dense2
#     nOut_trans2 = int(math.floor(nOut_dense2 * reduction))
#     nIn_dense3 = nOut_trans2
#     nOut_dense3 = nOut_trans2 + nDenseBlocks * growthRate
#     nIn_trans3 = nOut_dense3
#     nOut_trans3 = int(math.floor(nOut_dense3 * reduction))
#     nChannels = nOut_trans3
#     nOutChannels = gf_dim * 8
    
#     #model blocks
#     out = nn.Conv2d(c_dim, nOut_conv1, kernel_size=(3,3), bias=True, padding=(1,1), stride=(1, 1),)(x) ####
#     out_dwt0, out_dwt1, out_dwt2 = Dwtconv(nOut_trans1, nOut_trans2, nOut_trans3)(x)
#     out = _make_dense(nIn_dense1, opt.growthRate, nDenseBlocks, bottleneck)(out)
#     out = Transition(nIn_trans1, nOut_trans1)(out)
#     #print('out dwt1', out_dwt0.shape)
#     out = _make_dense(nIn_dense2, opt.growthRate, nDenseBlocks, bottleneck)(torch.add(out_dwt0, out))

#     out = Transition(nIn_trans2, nOut_trans2)(out)
#     out = make_dense(nIn_dense3, opt.growthRate, nDenseBlocks, bottleneck)(torch.add(out_dwt1, out))
#     out = Transition(nIn_trans3, nOut_trans3)(out)
#     out = nn.Conv2d(nChannels, nOutChannels, kernel_size=1)(nn.ReLU()(nn.BatchNorm2d(nChannels)(torch.add(out_dwt2, out))))
#     out = nn.ReLU()(nn.BatchNorm2d(nOutChannels)(out))
#     return out

# def Dwtconv(outC1, outC2, outC3):
#     from pytorch_wavelets import DWTForward
#     dwt1_1_l, dwt1_1_h = DWTForward(J=1, wave='haar', mode='symmetric')(x)
#     dwt1_1 = torch.cat((dwt1_1_l, dwt1_1_h[0][:,:,0], dwt1_1_h[0][:,:,1], dwt1_1_h[0][:,:,2]), dim=1)
#     outChannel_conv1 = outC1 // 2
#     nIn_conv1 = 4
#     conv1_1 = nn.Conv2d(nIn_conv1, outChannel_conv1, kernel_size=3, padding=1)(dwt1_1)
#     bn1_1 = nn.BatchNorm2d(outChannel_conv1)(conv1_1)
#     relu1_1 = nn.ReLU()(bn1_1)
#     conv1_2 = nn.Conv2d(outChannel_conv1, outC1, kernel_size=3, padding=1)(relu1_1)
#     bn1_2 = nn.BatchNorm2d(outC1)(conv1_2)
#     relu1_2 = nn.ReLU()(bn1_2)

#     nIn_conv2 = 4
#     outChannel_conv2 = outC2 // 2
#     dwt2_1_l, dwt2_1_h = DWTForward(J=1, wave='haar', mode='symmetric')(dwt1_1_l)
#     dwt2_1 = torch.cat((dwt2_1_l, dwt2_1_h[0][:, :, 0], dwt2_1_h[0][:, :, 1], dwt2_1_h[0][:, :, 2]), dim=1)
#     conv2_1 = nn.Conv2d(nIn_conv2, outChannel_conv2, kernel_size=3, padding=1)(dwt2_1)
#     bn2_1 = nn.BatchNorm2d(outChannel_conv2)(conv2_1)
#     relu2_1 = nn.ReLU()(bn2_1)
#     conv2_2 = nn.Conv2d(outChannel_conv2, outC2, kernel_size=3, padding=1)(relu2_1)
#     bn2_2 = nn.BatchNorm2d(outC2)(conv2_2)
#     relu2_2 = nn.ReLU()(bn2_2)

#     outChannel_conv3 = outC3 // 2
#     nIn_conv3 = 4
#     dwt3_1_l, dwt3_1_h = DWTForward(J=1, wave='haar', mode='symmetric')(dwt2_1_l)
#     dwt3_1 = torch.cat((dwt3_1_l, dwt3_1_h[0][:, :, 0], dwt3_1_h[0][:, :, 1], dwt3_1_h[0][:, :, 2]), dim=1)
#     conv3_1 = nn.Conv2d(nIn_conv3, outChannel_conv3, kernel_size=3, padding=1)(dwt3_1)
#     bn3_1 = nn.BatchNorm2d(outChannel_conv3)(conv3_1)
#     relu3_1 = nn.ReLU()(bn3_1)
#     conv3_2 = nn.Conv2d(outChannel_conv3, outC3, kernel_size=3, padding=1)(relu3_1)
#     bn3_2 = nn.BatchNorm2d(outC3)(conv3_2)
#     relu3_2 = nn.ReLU()(bn3_2)

#     return relu1_2, relu2_2, relu3_2

# # class ConvLstmCell(inputs, state, feature_size, num_features, gpu_ids, forget_bias=1,  bias=True):

# #     c, h = torch.chunk(state, 2, dim=1)
# #     conv_input = torch.cat((inputs, h), dim=1)
# #     conv_output = nn.Conv2d(num_features * 2, num_features * 4, feature_size, padding=int((feature_size - 1) / 2),
# #                               bias=bias)(conv_input)
# #     (i, j, f, o) = torch.chunk(conv_output, 4, dim=1)
# #     new_c = c * F.sigmoid(f + forget_bias) + F.sigmoid(i) * nn.Tanh()(j)
# #     new_h = nn.Tanh()(new_c) * F.sigmoid(o)
# #     new_state = torch.cat((new_c, new_h), dim=1)
# #     return new_h, new_state


# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
# #                 m.weight.data.normal_(0, math.sqrt(2./n))
# #             elif isinstance(m, nn.BatchNorm2d):
# #                 m.weight.data.fill_(1)
# #                 m.bias.data.zero_()
# #             elif isinstance(m, nn.Linear):
# #                 m.bias.data.zero_()


# def _make_dense(nChannels, growthRate, nDenseBlocks, bottleneck):
#     layers = []
#     for i in range(int(nDenseBlocks)):
#         if bottleneck:
#             layers.append(Bottleneck(nChannels, growthRate))
#         else:
#             layers.append(SingleLayer(nChannels, growthRate))
#         nChannels += growthRate
#     return nn.Sequential(*layers)

# def Bottleneck(x,nChannels, growthRate):
#     interChannels = 4 *growthRate 
#     out = nn.Conv2d(nChannels, interChannels, kernel_size=1)(nn.ReLU()(nn.BatchNorm2d(nChannels)(x)))
#     out = nn.Conv2d(interChannels, growthRate, kernel_size=3,padding=1)(nn.ReLU()(nn.BatchNorm2d(interChannels)(out)))
#     out = torch.cat((x, out), 1)
#     return out
        

# def SingleLayer(x,nChannels, growthRate):
#     out = nn.Conv2d(nChannels, growthRate, kernel_size=3,
#                                padding=1)(nn.ReLU()(nn.BatchNorm2d(nChannels)(x)))
#     out = torch.cat((x, out), 1)
#     return out

# def Transition(x,nChannels, nOutChannels):
#     out = nn.Conv2d(nChannels, nOutChannels, kernel_size=1)(nn.ReLU()(nn.BatchNorm2d(nChannels)(x)))
#     out = nn.AvgPool2d(2)(out)
#     return out

# def Decoder(x,c_dim, gf_dim, gpu_ids):
#     input3 = fixed_unpooling(x, gpu_ids)
#     deconv3_3 = nn.ConvTranspose2d(gf_dim * 8, gf_dim * 8, 3, padding=1)
#     relu3_3 = nn.ReLU()
#     deconv3_2 = nn.ConvTranspose2d(gf_dim * 8, gf_dim * 4, 3, padding=1)
#     relu3_2 = nn.ReLU()
#     deconv3_1 = nn.ConvTranspose2d(gf_dim * 4, gf_dim * 2, 3, padding=1)
#     relu3_1 = nn.ReLU()
#     dec3 = [deconv3_3, relu3_3, deconv3_2, relu3_2, deconv3_1, relu3_1]
#     dec3_out = nn.Sequential(*dec3)(input3)
    
#     input2 = fixed_unpooling(dec3_out, gpu_ids)
#     deconv2_2 = nn.ConvTranspose2d(gf_dim * 2, gf_dim * 2, 3, padding=1)
#     relu2_2 = nn.ReLU()
#     deconv2_1 = nn.ConvTranspose2d(gf_dim * 2, gf_dim, 3, padding=1)
#     relu2_1 = nn.ReLU()
#     dec2 = [deconv2_2, relu2_2, deconv2_1, relu2_1]
#     dec2_out = nn.Sequential(*dec2)(input2)
    
#     input1 = fixed_unpooling(dec2_out, gpu_ids)
#     deconv1_2 = nn.ConvTranspose2d(gf_dim, gf_dim, 3, padding=1)
#     relu1_2 = nn.ReLU()
#     deconv1_1 = nn.ConvTranspose2d(gf_dim, c_dim, 3, padding=1)
#     tanh1_1 = nn.Tanh()
#     dec1 = [deconv1_2, relu1_2, deconv1_1, tanh1_1]
#     dec1_out = nn.Sequential(*dec1)(input1)
#     return dec1_out

# from torch.autograd import Variable
# def fixed_unpooling(x, gpu_ids):
#     x = x.permute(0, 2, 3, 1)
#     out = torch.cat((x, Variable(torch.zeros(x.size()))), dim=3)
#     out = torch.cat((out, Variable(torch.zeros(out.size()))), dim=2)
#     sh = x.size()
#     s0, s1, s2, s3 = int(sh[0]), int(sh[1]), int(sh[2]), int(sh[3])
#     s1 *= 2
#     s2 *= 2
#     return out.view(s0, s1, s2, s3).permute(0, 3, 1, 2)

# def Generator(inputs,state,c_dim,K,T,batch_size,image_size):
#     encoder = Encoder(opt)
#     for k in range(K):
#         h_encoder = encoder(inputs[k])
#         h_dyn, state = convLstm_cell(h_encoder, state)
#     pred=[]
#     for t in range(T):
#         if t>0:
#             h_encoder = encoder(xt)
#             h_dyn, state = convLstm_cell(h_encoder, state)

#         x_hat = decoder(h_dyn)

#         xt = x_hat
#         pred.append(x_hat.view(batch_size, c_dim, image_size[0], image_size[1]))
#     return pred