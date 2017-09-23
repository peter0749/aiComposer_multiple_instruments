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
import midi
import random
import sys
import math
import h5py
import os.path
import argparse

parser = argparse.ArgumentParser(description='Music Generation with Mutiple Instruments (training)')
parser.add_argument('train_dir', metavar='training', type=str,
                    help='Path to the training set.')
parser.add_argument('valid_dir', metavar='validation', type=str,
                    help='Path to the validation set.')
parser.add_argument('--batch_size', type=int, default=128, required=False,
                    help='Number of samples per iteration.')
parser.add_argument('--epochs_inst', type=int, default=0, required=False,
                    help='Just a number of epochs. (train instrument predictor only)')
parser.add_argument('--epochs_note', type=int, default=0, required=False,
                    help='')
parser.add_argument('--epochs_delta', type=int, default=0, required=False,
                    help='')
parser.add_argument('--epochs', type=int, default=128, required=False,
                    help='Just a number of epochs. (will not train instrument classifier weights)')
parser.add_argument('--loop', type=int, default=1, required=False,
                    help='')
parser.add_argument('--sample_per_epoch', type=int, default=128, required=False,
                    help='Number of batchs every iteration.')
parser.add_argument('--lr', type=float, default=0.0001, required=False,
                    help='Learning rate.')
parser.add_argument('--no_drum', action='store_true', default=False,
                    help='No drums.')
parser.add_argument('--no_update_note', action='store_true', default=False,
                    help='')
parser.add_argument('--no_update_delta', action='store_true', default=False,
                    help='')
parser.add_argument('--no_update_inst', action='store_true', default=False,
                    help='')
parser.add_argument('--no_update_lstm', action='store_true', default=False,
                    help='')
parser.add_argument('--no_update_att', action='store_true', default=False,
                    help='')

args = parser.parse_args()

train_note = not args.no_update_note
train_delta= not args.no_update_delta
train_inst = not args.no_update_inst
train_lstm = not args.no_update_lstm
train_att  = not args.no_update_att

compute_precision='float32'
learning_rate = args.lr
loops = args.loop
epochs = args.epochs
epochs_inst = args.epochs_inst
epochs_note = args.epochs_note
epochs_delta = args.epochs_delta
samples_per_epoch = args.sample_per_epoch
no_drum = args.no_drum
step_size=1
segLen=48
vecLen=60 ## [36, 95]
maxdelta=33 ## [0, 32]
maxinst=129
batch_size = args.batch_size
hidden_delta=256
hidden_note=256
hidden_inst=256
filter_size=128
kernel_size=3 ## midi program changes are by groups
drop_rate=0.2 ## for powerful computer
Normal=120.0
defaultRes=16.0

K.set_floatx(compute_precision)

# Sorted x
def purge(x):
    y = [x[-1]] #pick last element
    i = len(x)-2 # iterate over all set in reversed order
    while i>=0:
        while i>=0 and x[i][1] == y[-1][1] and x[i][2] == y[-1][2]: ## overlapped, find next different note
            i -= 1
        if i>=0: ## founded
            y.append(x[i])
        i -= 1
    return list(reversed(y))
# merge some overlaped intervals into single interval

def Tempo2BPM(x):
    ret = x.data[2] | x.data[1]<<8 | x.data[0]<<16
    ret = float(60000000)/float(ret)
    return ret
## merge all tracks into one track

def pattern2map(pattern, maxtick):
    if len(pattern) > 1:
        return normal_pattern2map(pattern, maxtick)
    else:
        return ch_pattern2map(pattern, maxtick)

def normal_pattern2map(pattern, maxtick): ## tick range [0,63] #64
    ResScale = float(pattern.resolution) / float(defaultRes)
    data=[(0.0,0,0)]#tick , note, instrument, power
    instrument = 0 # sets to piano by default
    for track in pattern:
        temp=[(0.0,0,0)] #tick, note, instrument, power
        speedRatio = 1.0
        accumTick = 0.
        for v in track:
            if hasattr(v, 'tick') :
                accumTick = accumTick + float(v.tick)/speedRatio
            if isinstance(v, midi.ProgramChangeEvent):
                if hasattr(v, 'channel') and v.channel==9:
                    instrument = 128 # Percussion Key. a.k.a. drums
                else:
                    instrument = v.data[0]
            elif isinstance(v, midi.SetTempoEvent):
                changeBPM = Tempo2BPM(v)
                speedRatio = float(changeBPM)/float(Normal)
            elif isinstance(v, midi.NoteOnEvent) and v.data[0]>=36 and v.data[0]<=95 and v.data[1]>0:
                note = v.data[0]-36
                if instrument == 128 and no_drum:
                    continue
                temp.append((accumTick, note, instrument))
        temp = temp[1:-1]
        data.extend(temp)
    data = list(set(data)) ## remove duplicate data
    data.sort()
    data = purge(data)
    for i in range(0, len(data)-1):
        tick = data[i+1][0] - data[i][0]
        tick = int(round(tick/ResScale)) ## adjust resolution, downsampling
        tick = maxtick if tick>maxtick else tick ## set a threshold
        note = data[i+1][1]
        inst = data[i+1][2]
        data[i] = (tick,note,inst)
    data = data[0:-2] ## data must have two elements. ow crashs
    return data

def ch_pattern2map(pattern, maxtick): ## tick range [0,63] #64
    ch2ins = dict()
    ResScale = float(pattern.resolution) / float(defaultRes)
    data=[(0.0,0,0)]#tick , note, instrument, power
    instrument = 0 # sets to piano by default
    for track in pattern:
        temp=[(0.0,0,0)] #tick, note, instrument, power
        speedRatio = 1.0
        accumTick = 0.
        for v in track:
            if hasattr(v, 'tick') :
                accumTick = accumTick + float(v.tick)/speedRatio
            if isinstance(v, midi.ProgramChangeEvent):
                if hasattr(v, 'channel') and v.channel==9:
                    instrument = 128 # Percussion Key. a.k.a. drums
                else:
                    instrument = v.data[0]
                ch2ins[v.channel] = instrument
            elif isinstance(v, midi.SetTempoEvent):
                changeBPM = Tempo2BPM(v)
                speedRatio = float(changeBPM)/float(Normal)
            elif isinstance(v, midi.NoteOnEvent) and v.data[0]>=36 and v.data[0]<=95 and v.data[1]>0:
                ch = v.channel
                note = v.data[0]-36
                temp.append((accumTick, note, ch))
        temp = temp[1:-1]
        data.extend(temp)
    data = list(set(data)) ## remove duplicate data
    temp = []
    for i in xrange(len(data)): ## channel -> instrument
        acc = data[i][0]
        note= data[i][1]
        ch  = data[i][2]
        inst=ch2ins[ch]
        if inst==128 and no_drum:
            continue
        temp.append((acc,note,inst))
    data = temp
    temp = None
    data.sort() ## for better quality
    data = purge(data)
    for i in range(0, len(data)-1):
        tick = data[i+1][0] - data[i][0]
        tick = int(round(tick/ResScale)) ## adjust resolution, downsampling
        tick = maxtick if tick>maxtick else tick ## set a threshold
        note = data[i+1][1]
        inst = data[i+1][2]
        data[i] = (tick,note,inst)
    data = data[0:-2] ## data must have two elements. ow crashs
    return data

def makeSegment(data, maxlen, step):
    sentences = []
    nextseq = []
    for i in xrange(0, len(data) - maxlen, step):
        sentences.append(data[i: i + maxlen])
        nextseq.append(data[i + maxlen])
    return sentences, nextseq

def seg2vec(segment, nextseg, segLen, vecLen, maxdelta, maxinst):
    notes = np.zeros((len(segment), segLen, vecLen), dtype=np.bool)
    times = np.zeros((len(segment), segLen, maxdelta), dtype=np.bool)
    insts = np.zeros((len(segment), segLen, maxinst), dtype=np.bool)

    notes_n = np.zeros((len(segment), vecLen), dtype=np.bool)
    times_n = np.zeros((len(segment), maxdelta), dtype=np.bool)
    insts_n = np.zeros((len(segment), maxinst), dtype=np.bool)
    for i, seg in enumerate(segment):
        for t, note in enumerate(seg):
            times[i, t, int(note[0])] = 1
            notes[i, t, int(note[1])] = 1
            insts[i, t, int(note[2])] = 1
        times_n[i, int(nextseg[i][0])] = 1
        notes_n[i, int(nextseg[i][1])] = 1
        insts_n[i, int(nextseg[i][2])] = 1
    return notes, times, insts, notes_n, times_n, insts_n

def generator(path_name, step_size, batch_size, train_what=''):
    while True:
        randomFile = os.listdir(str(path_name))
        random.shuffle(randomFile)
        total_num = len(randomFile)
        for filename in randomFile:
            fullfile = str(path_name)+'/'+str(filename)
            try:
                pattern = midi.read_midifile(fullfile)
                data = pattern2map(pattern,maxdelta-1)
                seg, nextseg  = makeSegment(data, segLen, step_size)
                data = None ## clean-up
                note, time, inst, n_note, n_time, n_inst = seg2vec(seg, nextseg, segLen, vecLen, maxdelta, maxinst)
                seg = None
                nextseg = None
            except:
                pattern = data = seg = nextseg = None
                continue
            for i in xrange(0, len(note)-batch_size, batch_size):
                idx = range(i, i+batch_size)
                if train_what=='inst':
                    yield ([note[idx],time[idx],inst[idx]], [n_inst[idx]])
                elif train_what=='note':
                    yield ([note[idx],time[idx],inst[idx]], [n_note[idx]])
                elif train_what=='delta':
                    yield ([note[idx],time[idx],inst[idx]], [n_time[idx]])
                else: ## train all
                    yield ([note[idx],time[idx],inst[idx]], [n_note[idx],n_time[idx],n_inst[idx]])
            l = len(note)%batch_size
            if l > 0:
                idx = range(len(note)-l,len(note))
                if train_what=='inst':
                    yield ([note[idx],time[idx],inst[idx]], [n_inst[idx]])
                elif train_what=='note':
                    yield ([note[idx],time[idx],inst[idx]], [n_note[idx]])
                elif train_what=='delta':
                    yield ([note[idx],time[idx],inst[idx]], [n_time[idx]])
                else: ## train all
                    yield ([note[idx],time[idx],inst[idx]], [n_note[idx],n_time[idx],n_inst[idx]])

def main():
    # network:
    # build the model: stacked GRUs
    print('Build model...')
    # network:
    with tf.device('/gpu:0'):
        noteInput  = Input(shape=(segLen, vecLen))
        noteEncode = GRU(hidden_note, return_sequences=True, dropout=drop_rate, trainable=train_note)(noteInput)
        noteEncode = GRU(128, return_sequences=True, dropout=drop_rate, trainable=train_note)(noteEncode)

    with tf.device('/gpu:1'):
        deltaInput = Input(shape=(segLen, maxdelta))
        deltaEncode = GRU(hidden_delta, return_sequences=True, dropout=drop_rate, trainable=train_delta)(deltaInput)
        deltaEncode = GRU(128, return_sequences=True, dropout=drop_rate, trainable=train_delta)(deltaEncode)

    with tf.device('/gpu:2'):
        instInput = Input(shape=(segLen, maxinst))
        instEncode   = Conv1D(filters=filter_size, kernel_size=kernel_size, padding='same', input_shape=(segLen, maxinst), activation = 'relu', trainable=train_inst)(instInput)
        instEncode   = GRU(hidden_inst, return_sequences=True, dropout=drop_rate, trainable=train_inst)(instEncode)
        instEncode   = GRU(128, return_sequences=True, dropout=drop_rate, trainable=train_inst)(instEncode)

    with tf.device('/gpu:3'):
        codec = concatenate([noteEncode, deltaEncode, instEncode], axis=-1) ## return last state
        codec = SoftAttentionBlock(codec, segLen, 384, trainable=train_att)
        codec = LSTM(384, return_sequences=True, dropout=drop_rate, activation='softsign', trainable=train_lstm)(codec)
        codec = LSTM(256, return_sequences=False, dropout=drop_rate, activation='softsign', trainable=train_lstm)(codec)
        encoded = Dropout(drop_rate)(codec)

        fc_inst = BatchNormalization(trainable=train_inst)(encoded)
        pred_inst = Dense(maxinst, kernel_initializer='normal', activation='softmax', name='inst_output',  trainable=train_inst)(fc_inst) ## output PMF

        fc_notes = BatchNormalization(trainable=train_note)(encoded)
        pred_notes = Dense(vecLen, kernel_initializer='normal', activation='softmax', name='note_output', trainable=train_note)(fc_notes) ## output PMF

        fc_delta = BatchNormalization(trainable=train_delta)(encoded)
        pred_delta = Dense(maxdelta, kernel_initializer='normal', activation='softmax', name='time_output', trainable=train_delta)(fc_delta) ## output PMF
    aiComposer = Model([noteInput, deltaInput, instInput], [pred_notes, pred_delta, pred_inst])
    instClass  = Model([noteInput, deltaInput, instInput], [pred_inst])
    noteClass  = Model([noteInput, deltaInput, instInput], [pred_notes])
    deltaClass = Model([noteInput, deltaInput, instInput], [pred_delta])
    checkPoint = ModelCheckpoint(filepath="top_weight.h5", verbose=1, save_best_only=True, save_weights_only=True, period=1)
    instCheckPoint = ModelCheckpoint(filepath="inst_weight.h5", verbose=1, save_best_only=False, save_weights_only=True, period=1)
    noteCheckPoint = ModelCheckpoint(filepath="note_weight.h5", verbose=1, save_best_only=False, save_weights_only=True, period=1)
    deltaCheckPoint = ModelCheckpoint(filepath="delta_weight.h5", verbose=1, save_best_only=False, save_weights_only=True, period=1)
    Logs = CSVLogger('logs.csv', separator=',', append=True)
    instLogs = CSVLogger('logs_inst.csv', separator=',', append=True)
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
                             'inst_output':1,
                             'note_output':1e-1,
                             'time_output':1e-2
                           },
            optimizer=optimizer, metrics=['accuracy'])
    instClass.compile(
            loss = 'categorical_crossentropy',
            optimizer=optimizer, metrics=['accuracy'])
    noteClass.compile(
            loss = 'categorical_crossentropy',
            optimizer=optimizer, metrics=['accuracy'])
    deltaClass.compile(
            loss = 'categorical_crossentropy',
            optimizer=optimizer, metrics=['accuracy'])

    ## layer sets:
    full_dict = dict([(layer.name, layer) for layer in aiComposer.layers])
    inst_dict = dict([(layer.name, layer) for layer in instClass.layers])
    note_dict = dict([(layer.name, layer) for layer in noteClass.layers])
    delta_dict = dict([(layer.name, layer) for layer in deltaClass.layers])

    ## set weights:
    for layer_name in full_dict:
        if layer_name in inst_dict:
            inst_dict[layer_name].set_weights(full_dict[layer_name].get_weights())
        if layer_name in note_dict:
            note_dict[layer_name].set_weights(full_dict[layer_name].get_weights())
        if layer_name in delta_dict:
            delta_dict[layer_name].set_weights(full_dict[layer_name].get_weights())

    for ite in xrange(loops):
        if epochs_inst>0:
            instClass.fit_generator(generator(args.train_dir, step_size, batch_size, 'inst'), steps_per_epoch=samples_per_epoch, epochs=epochs_inst, validation_data=generator(args.valid_dir, step_size, batch_size, 'inst'), validation_steps=5, callbacks=[instCheckPoint, instLogs]) ## fine tune instrument classifier
            for l in inst_dict: full_dict[l].set_weights(inst_dict[l].get_weights())
        if epochs_note>0:
            noteClass.fit_generator(generator(args.train_dir, step_size, batch_size, 'note'), steps_per_epoch=samples_per_epoch, epochs=epochs_note, validation_data=generator(args.valid_dir, step_size, batch_size, 'note'), validation_steps=5, callbacks=[noteCheckPoint, noteLogs]) ## fine tune note classifier
            for l in note_dict: full_dict[l].set_weights(note_dict[l].get_weights())
        if epochs_delta>0:
            deltaClass.fit_generator(generator(args.train_dir, step_size, batch_size, 'delta'), steps_per_epoch=samples_per_epoch, epochs=epochs_delta, validation_data=generator(args.valid_dir, step_size, batch_size, 'delta'), validation_steps=5, callbacks=[deltaCheckPoint, deltaLogs]) ## fine tune tick classifier
            for l in delta_dict: full_dict[l].set_weights(delta_dict[l].get_weights())
        aiComposer.fit_generator(generator(args.train_dir, step_size, batch_size, 'all'), steps_per_epoch=samples_per_epoch, epochs=epochs, validation_data=generator(args.valid_dir, step_size, batch_size, 'all'), validation_steps=10, callbacks=[checkPoint, Logs])
        aiComposer.save('./multi-%d.h5' % ite)
    aiComposer.save('./multi.h5')

if __name__ == "__main__":
    main()
