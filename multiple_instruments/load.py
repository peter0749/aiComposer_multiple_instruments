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
track_num=2
vecLen=60*track_num ## two tracks
maxdelta=33 ## [0,32]
maxpower=64
batch_size=1
hidden_delta=256
hidden_note=256
hidden_inst=256
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

with tf.device('/gpu:3'):
    codec = concatenate([noteEncode, deltaEncode], axis=-1) ## return last state
    codec = SoftAttentionBlock(codec, segLen, 256)
    codec = LSTM(256, return_sequences=True, dropout=drop_rate, activation='softsign')(codec)
    codec = LSTM(256, return_sequences=False, dropout=drop_rate, activation='softsign')(codec)
    encoded = Dropout(drop_rate)(codec)

    fc_notes = BatchNormalization()(encoded)
    pred_notes = Dense(vecLen, kernel_initializer='normal', activation='softmax', name='note_output')(fc_notes) ## output PMF

    fc_delta = BatchNormalization()(encoded)
    pred_delta = Dense(maxdelta, kernel_initializer='normal', activation='softmax', name='time_output')(fc_delta) ## output PMF

aiComposer = Model([noteInput, deltaInput], [pred_notes, pred_delta])
if ( os.path.isfile('./top_weight.h5')):  ## fine-tuning
    aiComposer.load_weights('./top_weight.h5')
aiComposer.save('./multi.h5')
