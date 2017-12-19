from __future__ import print_function
import os
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, Conv1D
from keras.layers import LSTM, RepeatVector, TimeDistributed, BatchNormalization
from keras.layers.merge import concatenate
from keras import regularizers
from keras.optimizers import RMSprop
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import backend as K
import numpy as np
import midi
import random
import sys
import math
import h5py
import os.path
import argparse
from tools import *

parser = argparse.ArgumentParser(description='Music Generation with Mutiple Instruments (training)')
parser.add_argument('train_dir', metavar='training', type=str,
                    help='Path to the training set.')
parser.add_argument('valid_dir', metavar='validation', type=str,
                    help='Path to the validation set.')
parser.add_argument('--batch_size', type=int, default=128, required=False,
                    help='Number of samples per iteration.')
parser.add_argument('--epochs', type=int, default=128, required=False,
                    help='Just a number of epochs.')
parser.add_argument('--loop', type=int, default=1, required=False,
                    help='')
parser.add_argument('--sample_per_epoch', type=int, default=128, required=False,
                    help='Number of batchs every iteration.')
parser.add_argument('--lr', type=float, default=0.0001, required=False,
                    help='Learning rate.')
parser.add_argument('--not_do_shift', action='store_true', default=False,
                    help='')

args = parser.parse_args()

not_do_shift = args.not_do_shift

compute_precision='float32'
learning_rate = args.lr
loops = args.loop
epochs = args.epochs
samples_per_epoch = args.sample_per_epoch
step_size=1
segLen=48
track_num=1
maxrange=60 ## [36, 95]
maxvol=32
vecLen=maxrange*track_num
maxdelta=33 ## [0, 32]
batch_size = args.batch_size
hidden_delta=128
hidden_note=256
drop_rate=0.14 ## for powerful computer
Normal=120.0
defaultRes=16.0

K.set_floatx(compute_precision)

def generator(path_name, step_size, batch_size, valid=False):
    while True:
        randomFile = os.listdir(str(path_name))
        random.shuffle(randomFile)
        total_num = len(randomFile)
        for filename in randomFile:
            fullfile = str(path_name)+'/'+str(filename)
            try:
                pattern = midi.read_midifile(fullfile)
                #if pattern.format != 1: continue
                data = pattern2map(pattern,maxdelta-1)
                seg, nextseg  = makeSegment(data, segLen, step_size, valid)
                del data ## clean-up
                note, time, power, __, _, n_power = seg2vec(seg, nextseg, segLen, vecLen, maxdelta)
                del seg
                del nextseg
                del _
                del __
            except:
                sys.stderr.write('something wrong...:\\')
                continue
            for i in xrange(0, len(note)-batch_size, batch_size):
                idx = range(i, i+batch_size)
                yield ([note[idx],time[idx], power[idx]], n_power[idx])
            l = len(note)%batch_size
            if l > 0:
                idx = range(len(note)-l,len(note))
                yield ([note[idx],time[idx], power[idx]], n_power[idx])

def main():
    # network:
    # build the model: stacked LSTMs
    print('Build model...')
    # network:
    noteInput  = Input(shape=(segLen, vecLen))
    deltaInput = Input(shape=(segLen, maxdelta))
    volInput = Input(shape=(segLen, maxvol))

    c1 = concatenate([noteInput, deltaInput, volInput], axis=-1)
    c1 = Flatten()(c1)
    c1 = Dropout(drop_rate)(c1)
    fc1 = Dense(128, activation='relu')(c1)
    fc2 = Dropout(drop_rate)(fc1)
    fc_vol = Dense(maxvol)(fc2)
    fc_vol = BatchNormalization()(fc_vol)
    fc_vol = Activation('softmax')(fc_vol)

    aiComposer = Model([noteInput, deltaInput, volInput], fc_vol)
    checkPoint = ModelCheckpoint(filepath="top_weight.h5", verbose=1, save_best_only=True, save_weights_only=True, period=1)
    Logs = CSVLogger('logs.csv', separator=',', append=True)
    optimizer = RMSprop(lr=learning_rate, decay=learning_rate/epochs, clipnorm=1.)
    if ( os.path.isfile('./top_weight.h5')):  ## fine-tuning
        aiComposer.load_weights('./top_weight.h5')
    aiComposer.summary()

    ## compile models:
    aiComposer.compile(
            loss = 'categorical_crossentropy',
            optimizer=optimizer,
            metrics=['acc']
            )

    for ite in xrange(loops):
        aiComposer.fit_generator(generator(args.train_dir, step_size, batch_size, valid=not_do_shift), steps_per_epoch=samples_per_epoch, epochs=epochs, validation_data=generator(args.valid_dir, step_size, batch_size, valid=True), validation_steps=80, callbacks=[checkPoint, Logs])
        aiComposer.save('./velocity-%d.h5' % ite)
    aiComposer.save('./velocity.h5')

if __name__ == "__main__":
    main()
