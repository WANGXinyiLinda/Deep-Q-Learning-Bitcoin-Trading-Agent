from __future__ import division

import six # Python 2 and 3 compatibility library
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout,
    GRU
)
from keras.layers.convolutional import (
    Conv1D,
    MaxPooling1D,
    AveragePooling1D
)
from keras.models import Sequential
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
import numpy as np

def _handle_dim_ordering():
    global BATCH_SIZE
    global LENGTH_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        BATCH_SIZE = 0
        LENGTH_AXIS = 1
        CHANNEL_AXIS = 2
    else:
        BATCH_SIZE = 0
        CHANNEL_AXIS = 1
        LENGTH_AXIS = 2


class modelBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs):
        _handle_dim_ordering()
        if len(input_shape) != 2:
            raise Exception("Input shape should be a tuple (nb_channels, length)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[0])

        model = Sequential()

        model.add(Conv1D(32, 5, activation='relu', input_shape= input_shape))
        model.add(Dropout(0.1))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(Dropout(0.1))
        model.add(GRU(units = 128, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(GRU(units = 128))
        model.add(Dropout(0.5))
        model.add(Dense(units = 64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units = 32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=num_outputs, activation='linear'))

        return model
