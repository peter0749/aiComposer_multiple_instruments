from __future__ import print_function
import os
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, Conv1D
from keras.layers import LSTM, RepeatVector, TimeDistributed
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import backend as K
import numpy as np
import sys
import math
import h5py
import os.path

compute_precision='float32'
learning_rate = 0.002
epochs = 1
segLen=48
track_num=1
vecLen=60*track_num ## two tracks
maxdelta=33 ## [0,32]
maxpower=64
batch_size=1
hidden_delta=128
hidden_note=256
hidden_inst=256
drop_rate=0.2 ## for powerful computer

K.set_floatx(compute_precision);

# build the model: stacked LSTMs
print('Build model...')
# network:
with tf.device('/gpu:0'):
    noteInput  = Input(shape=(segLen, vecLen))

with tf.device('/gpu:1'):
    deltaInput = Input(shape=(segLen, maxdelta))

with tf.device('/gpu:3'):
    codec = concatenate([noteInput, deltaInput], axis=-1)
    codec = LSTM(600, return_sequences=True)(codec)
    codec = Dropout(drop_rate)(codec)
    codec = LSTM(600, return_sequences=True)(codec)
    codec = Dropout(drop_rate)(codec)
    codec = LSTM(600, return_sequences=False)(codec)
    encoded = Dropout(drop_rate)(codec)

    fc_notes = Dense(vecLen, kernel_initializer='normal')(encoded) ## output PMF
    pred_notes = Activation('softmax', name='note_output')(fc_notes)

    fc_delta = Dense(maxdelta, kernel_initializer='normal')(encoded) ## output PMF
    pred_delta = Activation('softmax', name='time_output')(fc_delta) ## output PMF

aiComposer = Model([noteInput, deltaInput], [pred_notes, pred_delta])
aiComposer.summary()
from keras.utils import plot_model
plot_model(aiComposer, to_file='model.png', show_shapes=True)

