from __future__ import print_function
import os
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, Conv1D
from keras.layers import LSTM, GRU, BatchNormalization, RepeatVector, TimeDistributed
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import backend as K
from attention_block import SoftAttentionBlock
import numpy as np
import sys
import math
import h5py
import os.path

compute_precision='float32'
learning_rate = 0.002
epochs = 1
segLen=48
vecLen=36
maxdelta=32
maxinst=129
maxpower=64
batch_size=1
hidden_delta=256
hidden_note=256
hidden_inst=256
filter_size=128
kernel_size=3 ## midi program changes are by groups
drop_rate=0.2 ## for powerful computer

K.set_floatx(compute_precision);

# build the model: stacked GRUs
print('Build model...')
# network:
with tf.device('/gpu:0'):
    noteInput  = Input(shape=(segLen, vecLen))
    noteEncode = GRU(hidden_note, return_sequences=True, dropout=drop_rate)(noteInput)
    noteEncode = GRU(128, return_sequences=True, dropout=drop_rate)(noteEncode)

with tf.device('/gpu:1'):
    deltaInput = Input(shape=(segLen, maxdelta))
    deltaEncode = GRU(hidden_delta, return_sequences=True, dropout=drop_rate)(deltaInput)
    deltaEncode = GRU(128, return_sequences=True, dropout=drop_rate)(deltaEncode)

with tf.device('/gpu:2'):
    instInput = Input(shape=(segLen, maxinst))
    instEncode   = Conv1D(filters=filter_size, kernel_size=kernel_size, padding='same', input_shape=(segLen, maxinst), activation = 'relu')(instInput)
    instEncode   = GRU(hidden_inst, return_sequences=True, dropout=drop_rate)(instEncode)
    instEncode   = GRU(128, return_sequences=True, dropout=drop_rate)(instEncode)

with tf.device('/gpu:3'):
    codec = concatenate([noteEncode, deltaEncode, instEncode], axis=-1) ## return last state
    codec = SoftAttentionBlock(codec, segLen, 384)
    codec = LSTM(384, return_sequences=True, dropout=drop_rate, activation='softsign')(codec)
    codec = LSTM(256, return_sequences=False, dropout=drop_rate, activation='softsign')(codec)
    encoded = Dropout(drop_rate)(codec)
    fc_inst = BatchNormalization()(encoded)
    pred_inst = Dense(maxinst, kernel_initializer='normal', activation='softmax', name='inst_output')(fc_inst) ## output PMF
    pred_inst_reduce = Dense(3, kernel_initializer='normal')(pred_inst)
    arg_feature   = concatenate([encoded, pred_inst_reduce], axis=-1)

    fc_notes = BatchNormalization()(arg_feature)
    pred_notes = Dense(vecLen, kernel_initializer='normal', activation='softmax', name='note_output')(fc_notes) ## output PMF

    fc_delta = BatchNormalization()(arg_feature)
    pred_delta = Dense(maxdelta, kernel_initializer='normal', activation='softmax', name='time_output')(fc_delta) ## output PMF

    fc_power = BatchNormalization()(arg_feature)
    pred_power = Dense(1, kernel_initializer='normal', activation='relu', name='power_output')(fc_power) ## output regression >= 0
aiComposer = Model([noteInput, deltaInput, instInput], [pred_notes, pred_delta, pred_inst, pred_power])
if ( os.path.isfile('./top_weight.h5')):  ## fine-tuning
    aiComposer.load_weights('./top_weight.h5')
aiComposer.save('./multi.h5')
