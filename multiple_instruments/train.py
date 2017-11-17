from __future__ import print_function
import os
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, Conv1D
from keras.layers import CuDNNLSTM, RepeatVector, TimeDistributed
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
parser.add_argument('--epochs_note', type=int, default=0, required=False,
                    help='')
parser.add_argument('--epochs_delta', type=int, default=0, required=False,
                    help='')
parser.add_argument('--epochs', type=int, default=128, required=False,
                    help='Just a number of epochs.')
parser.add_argument('--loop', type=int, default=1, required=False,
                    help='')
parser.add_argument('--sample_per_epoch', type=int, default=128, required=False,
                    help='Number of batchs every iteration.')
parser.add_argument('--lr', type=float, default=0.0001, required=False,
                    help='Learning rate.')
parser.add_argument('--no_update_note', action='store_true', default=False,
                    help='')
parser.add_argument('--no_update_delta', action='store_true', default=False,
                    help='')
parser.add_argument('--no_update_lstm', action='store_true', default=False,
                    help='')
parser.add_argument('--no_update_att', action='store_true', default=False,
                    help='')
parser.add_argument('--not_do_shift', action='store_true', default=False,
                    help='')

args = parser.parse_args()

train_note = not args.no_update_note
train_delta= not args.no_update_delta
train_lstm = not args.no_update_lstm
train_att  = not args.no_update_att
not_do_shift = args.not_do_shift

compute_precision='float32'
learning_rate = args.lr
loops = args.loop
epochs = args.epochs
epochs_note = args.epochs_note
epochs_delta = args.epochs_delta
samples_per_epoch = args.sample_per_epoch
step_size=1
segLen=48
track_num=1
maxrange=60 ## [36, 95]
vecLen=maxrange*track_num
maxdelta=33 ## [0, 32]
batch_size = args.batch_size
hidden_delta=128
hidden_note=256
drop_rate=0.14 ## for powerful computer
Normal=120.0
defaultRes=16.0

K.set_floatx(compute_precision)

def generator(path_name, step_size, batch_size, train_what='', valid=False):
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
                note, time, n_note, n_time = seg2vec(seg, nextseg, segLen, vecLen, maxdelta)
                del seg
                del nextseg
            except:
                sys.stderr.write('something wrong...:\\')
                continue
            for i in xrange(0, len(note)-batch_size, batch_size):
                idx = range(i, i+batch_size)
                if train_what=='note':
                    yield ([note[idx],time[idx]], [n_note[idx]])
                elif train_what=='delta':
                    yield ([note[idx],time[idx]], [n_time[idx]])
                else: ## train all
                    yield ([note[idx],time[idx]], [n_note[idx],n_time[idx]])
            l = len(note)%batch_size
            if l > 0:
                idx = range(len(note)-l,len(note))
                if train_what=='note':
                    yield ([note[idx],time[idx]], [n_note[idx]])
                elif train_what=='delta':
                    yield ([note[idx],time[idx]], [n_time[idx]])
                else: ## train all
                    yield ([note[idx],time[idx]], [n_note[idx],n_time[idx]])

def main():
    # network:
    # build the model: stacked LSTMs
    print('Build model...')
    # network:
    noteInput  = Input(shape=(segLen, vecLen))

    deltaInput = Input(shape=(segLen, maxdelta))

    codec = concatenate([noteInput, deltaInput], axis=-1) ## return last state
    codec = CuDNNLSTM(600, return_sequences=True, unit_forget_bias=True, recurrent_regularizer=regularizers.l2(5e-4), bias_regularizer=regularizers.l1(5e-4), trainable=train_lstm)(codec)
    codec = Dropout(drop_rate)(codec)
    codec = CuDNNLSTM(600, return_sequences=True, unit_forget_bias=True, recurrent_regularizer=regularizers.l2(2e-4), bias_regularizer=regularizers.l1(2e-4), trainable=train_lstm)(codec)
    codec = Dropout(drop_rate)(codec)
    codec = CuDNNLSTM(600, return_sequences=False, unit_forget_bias=True, recurrent_regularizer=regularizers.l2(1e-4), bias_regularizer=regularizers.l1(1e-4), trainable=train_lstm)(codec)
    encoded = Dropout(drop_rate)(codec)

    fc_notes = Dense(vecLen, kernel_initializer='normal', trainable=train_note)(encoded) ## output PMF
    pred_notes = Activation('softmax', name='note_output', trainable=train_note)(fc_notes)

    fc_delta = Dense(maxdelta, kernel_initializer='normal', trainable=train_delta)(encoded) ## output PMF
    pred_delta = Activation('softmax', name='time_output', trainable=train_delta)(fc_delta) ## output PMF

    aiComposer = Model([noteInput, deltaInput], [pred_notes, pred_delta])
    noteClass  = Model([noteInput, deltaInput], [pred_notes])
    deltaClass = Model([noteInput, deltaInput], [pred_delta])
    checkPoint = ModelCheckpoint(filepath="top_weight.h5", verbose=1, save_best_only=True, save_weights_only=True, period=1)
    noteCheckPoint = ModelCheckpoint(filepath="note_weight.h5", verbose=1, save_best_only=False, save_weights_only=True, period=1)
    deltaCheckPoint = ModelCheckpoint(filepath="delta_weight.h5", verbose=1, save_best_only=False, save_weights_only=True, period=1)
    Logs = CSVLogger('logs.csv', separator=',', append=True)
    noteLogs = CSVLogger('logs_note.csv', separator=',', append=True)
    deltaLogs = CSVLogger('logs_delta.csv', separator=',', append=True)
    optimizer = RMSprop(lr=learning_rate, decay=learning_rate/epochs, clipnorm=1.)
    if ( os.path.isfile('./top_weight.h5')):  ## fine-tuning
        aiComposer.load_weights('./top_weight.h5')
    aiComposer.summary()

    ## compile models:
    aiComposer.compile(
            loss = 'categorical_crossentropy',
            loss_weights = {
                             'note_output':1,
                             'time_output':1e-1
                           },
            optimizer=optimizer, metrics=['accuracy'])
    noteClass.compile(
            loss = 'categorical_crossentropy',
            optimizer=optimizer, metrics=['accuracy'])
    deltaClass.compile(
            loss = 'categorical_crossentropy',
            optimizer=optimizer, metrics=['accuracy'])

    ## layer sets:
    full_dict = dict([(layer.name, layer) for layer in aiComposer.layers])
    note_dict = dict([(layer.name, layer) for layer in noteClass.layers])
    delta_dict = dict([(layer.name, layer) for layer in deltaClass.layers])

    ## set weights:
    for layer_name in full_dict:
        if layer_name in note_dict:
            note_dict[layer_name].set_weights(full_dict[layer_name].get_weights())
        if layer_name in delta_dict:
            delta_dict[layer_name].set_weights(full_dict[layer_name].get_weights())

    for ite in xrange(loops):
        if epochs_note>0:
            noteClass.fit_generator(generator(args.train_dir, step_size, batch_size, 'note', valid=not_do_shift), steps_per_epoch=samples_per_epoch, epochs=epochs_note, validation_data=generator(args.valid_dir, step_size, batch_size, 'note', valid=True), validation_steps=5, callbacks=[noteCheckPoint, noteLogs]) ## fine tune note classifier
            for l in note_dict: full_dict[l].set_weights(note_dict[l].get_weights())
        if epochs_delta>0:
            deltaClass.fit_generator(generator(args.train_dir, step_size, batch_size, 'delta', valid=not_do_shift), steps_per_epoch=samples_per_epoch, epochs=epochs_delta, validation_data=generator(args.valid_dir, step_size, batch_size, 'delta', valid=True), validation_steps=5, callbacks=[deltaCheckPoint, deltaLogs]) ## fine tune tick classifier
            for l in delta_dict: full_dict[l].set_weights(delta_dict[l].get_weights())
        aiComposer.fit_generator(generator(args.train_dir, step_size, batch_size, 'all', valid=not_do_shift), steps_per_epoch=samples_per_epoch, epochs=epochs, validation_data=generator(args.valid_dir, step_size, batch_size, 'all', valid=True), validation_steps=10, callbacks=[checkPoint, Logs])
        aiComposer.save('./multi-%d.h5' % ite)
    aiComposer.save('./multi.h5')

if __name__ == "__main__":
    main()
