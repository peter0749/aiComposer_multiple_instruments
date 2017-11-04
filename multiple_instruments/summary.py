from __future__ import print_function
import os
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, Conv1D
from keras.layers import GRU, LSTM, BatchNormalization, RepeatVector, TimeDistributed
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
segLen=100
vecLen=60
maxdelta=33 ## [0,32]
maxinst=14
maxpower=64
batch_size=1
hidden_delta=128
hidden_note=128
hidden_inst=128
filter_size=128
kernel_size=3 ## midi program changes are by groups
drop_rate=0.2 ## for powerful computer

K.set_floatx(compute_precision);

# build the model: stacked LSTMs
print('Build model...')
# network:
with tf.device('/gpu:0'):
    noteInput  = Input(shape=(segLen, vecLen))
    noteEncode = GRU(hidden_note, return_sequences=True, dropout=drop_rate)(noteInput)

with tf.device('/gpu:1'):
    deltaInput = Input(shape=(segLen, maxdelta))
    deltaEncode = GRU(hidden_delta, return_sequences=True, dropout=drop_rate)(deltaInput)

with tf.device('/gpu:2'):
    instInput = Input(shape=(segLen, maxinst))
    instEncode   = GRU(hidden_inst, return_sequences=True, dropout=drop_rate)(instInput)

with tf.device('/gpu:3'):
    codec = concatenate([noteEncode, deltaEncode, instEncode], axis=-1) ## return last state
    codec = LSTM(600, return_sequences=True, dropout=drop_rate, activation='tanh')(codec)
    codec = LSTM(600, return_sequences=True, dropout=drop_rate, activation='tanh')(codec)
    codec = LSTM(600, return_sequences=False, dropout=drop_rate, activation='tanh')(codec)
    encoded = Dropout(drop_rate)(codec)

    fc_inst = Dense(maxinst, kernel_initializer='normal')(encoded)
    fc_inst = BatchNormalization()(fc_inst)
    pred_inst = Activation('softmax', name='inst_output')(fc_inst)

    fc_notes = Dense(vecLen, kernel_initializer='normal')(encoded) ## output PMF
    fc_notes = BatchNormalization()(fc_notes)
    pred_notes = Activation('softmax', name='note_output')(fc_notes)

    fc_delta = Dense(maxdelta, kernel_initializer='normal')(encoded) ## output PMF
    fc_delta = BatchNormalization()(fc_delta)
    pred_delta = Activation('softmax', name='time_output')(fc_delta) ## output PMF

aiComposer = Model([noteInput, deltaInput, instInput], [pred_notes, pred_delta, pred_inst])
aiComposer.summary()
from keras.utils import plot_model
plot_model(aiComposer, to_file='model.png', show_shapes=True)

