from __future__ import division

from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, Add


def basic_block(X, filters, is_first_layer, is_first_block):

    strides = (1, 1) if is_first_layer or not is_first_block else (2, 2)
    X_input = X

    if is_first_layer and is_first_block:
        X = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same')(X)
    else:
        X = _block_layers(X, filters, strides)

    X = _block_layers(X, filters, strides=(1, 1))
    return _shortcut(X_input, X)


def _block_layers(X, filters, strides):
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    return Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same')(X)


def _shortcut(X, residual):
    input_shape = K.int_shape(X)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]
    shortcut = X

    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3], kernel_size=(1, 1), strides=(stride_width, stride_height))(X)

    return Add()([shortcut, residual])
