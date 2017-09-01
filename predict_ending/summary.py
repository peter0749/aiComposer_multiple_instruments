from __future__ import print_function
import os
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, Conv1D
from keras.layers import LSTM, BatchNormalization, RepeatVector, TimeDistributed, Bidirectional
from keras.layers.merge import concatenate
from keras.layers.core import Masking
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
segLen=128
endLen=64
vecLen=88
maxdelta=128
maxinst=129
maxpower=64
batch_size=1
hidden_delta=128
hidden_note=128
hidden_inst=128
hidden_decoder=512
filter_size=128
kernel_size=7 ## midi program changes are by groups
drop_rate=0.2 ## for powerful computer

K.set_floatx(compute_precision);

# build the model: stacked LSTMs
print('Build model...')
# network:
with tf.device('/gpu:0'):
    noteInput  = Input(shape=(segLen, vecLen))
    noteEncode = Bidirectional(LSTM(hidden_note, input_shape=(segLen, vecLen), return_sequences=True, dropout=drop_rate))(noteInput)
    noteEncode = Bidirectional(LSTM(hidden_note, return_sequences=False, dropout=drop_rate))(noteEncode)

with tf.device('/gpu:1'):
    deltaInput = Input(shape=(segLen, maxdelta))
    deltaEncode = Bidirectional(LSTM(hidden_delta, input_shape=(segLen, maxdelta), return_sequences=True, dropout=drop_rate))(deltaInput)
    deltaEncode = Bidirectional(LSTM(hidden_delta, return_sequences=False, dropout=drop_rate))(deltaEncode)

with tf.device('/gpu:2'):
    instInput = Input(shape=(segLen, maxinst))
    instEncode   = Conv1D(filters=filter_size, kernel_size=kernel_size, padding='same', input_shape=(segLen, maxinst), activation = 'relu')(instInput)
    instEncode   = Bidirectional(LSTM(hidden_inst, return_sequences=True, dropout=drop_rate))(instEncode)
    instEncode   = Bidirectional(LSTM(hidden_inst, return_sequences=False, dropout=drop_rate))(instEncode)

with tf.device('/gpu:3'):
    encoder = concatenate([noteEncode, deltaEncode, instEncode], axis=-1)
    decoder = RepeatVector(endLen)(encoder)
    decoder = LSTM(hidden_decoder, return_sequences=True, dropout=drop_rate)(decoder)
    decoder = LSTM(hidden_decoder, return_sequences=True, dropout=drop_rate)(decoder)

    batchNormNote_0 = TimeDistributed(BatchNormalization())(decoder)
    fc_notes = TimeDistributed(Dense(vecLen*2, kernel_initializer='normal', activation='relu'))(batchNormNote_0)
    fc_notes = TimeDistributed(BatchNormalization())(fc_notes)
    pred_notes = TimeDistributed(Dense(vecLen, kernel_initializer='normal', activation='softmax'), name='note_output')(fc_notes) ## output PMF

    batchNormDelta_0 = TimeDistributed(BatchNormalization())(decoder)
    fc_delta = TimeDistributed(Dense(maxdelta*2, kernel_initializer='normal', activation='relu'))(batchNormDelta_0)
    fc_delta = TimeDistributed(BatchNormalization())(fc_delta)
    pred_delta = TimeDistributed(Dense(maxdelta, kernel_initializer='normal', activation='softmax'), name='time_output')(fc_delta) ## output PMF

    batchNormInst_0 = TimeDistributed(BatchNormalization())(decoder)
    fc_inst = TimeDistributed(Dense(maxinst*2, kernel_initializer='normal', activation='relu'))(batchNormInst_0)
    fc_inst = TimeDistributed(BatchNormalization())(fc_inst)
    pred_inst = TimeDistributed(Dense(maxinst, kernel_initializer='normal', activation='softmax'), name='inst_output')(fc_inst) ## output PMF

    batchNormPower_0 = TimeDistributed(BatchNormalization())(decoder)
    fc_power = TimeDistributed(Dense(maxpower, kernel_initializer='normal', activation='relu'))(batchNormPower_0)
    fc_power = TimeDistributed(BatchNormalization())(fc_power)
    pred_power = TimeDistributed(Dense(1, kernel_initializer='normal', activation='relu'), name='power_output')(fc_power) ## output regression >= 0
aiComposer = Model([noteInput, deltaInput, instInput], [pred_notes, pred_delta, pred_inst, pred_power])
aiComposer.summary()
from keras.utils import plot_model
plot_model(aiComposer, to_file='model.png', show_shapes=True)
