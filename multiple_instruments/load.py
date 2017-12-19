from __future__ import print_function
import os
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, Conv1D
from keras.layers import CuDNNLSTM, RepeatVector, TimeDistributed, BatchNormalization
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
maxvol=32
batch_size=1
hidden_delta=128
hidden_note=256
hidden_inst=256
drop_rate=0.2 ## for powerful computer

K.set_floatx(compute_precision);

# build the model: stacked LSTMs
print('Build model...')
# network:
noteInput  = Input(shape=(segLen, vecLen))
deltaInput = Input(shape=(segLen, maxdelta))
volInput = Input(shape=(segLen, maxvol))

c1 = concatenate([noteInput, deltaInput, volInput], axis=-1)
fc1 = CuDNNLSTM(128, return_sequences=False, unit_forget_bias=True)(c1)
fc2 = Dropout(drop_rate)(fc1)
fc_vol = Dense(maxvol)(fc2)
fc_vol = BatchNormalization()(fc_vol)
fc_vol = Activation('softmax')(fc_vol)

aiComposer = Model([noteInput, deltaInput, volInput], fc_vol)
if ( os.path.isfile('./top_weight.h5')):  ## fine-tuning
    aiComposer.load_weights('./top_weight.h5')
aiComposer.save('./velocity.h5')
