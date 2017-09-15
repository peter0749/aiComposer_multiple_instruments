from __future__ import print_function
import os
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, Conv1D
from keras.layers import GRU, BatchNormalization, RepeatVector, TimeDistributed
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import backend as K
from attention_block import SoftAttentionBlock, Attention
import numpy as np
import midi
import random
import sys
import math
import h5py
import os.path

compute_precision='float32'
learning_rate = 0.0002
epochs = int(sys.argv[4])
samples_per_epoch = int(sys.argv[5])
step_size=1
segLen=128
vecLen=88
maxdelta=128
maxinst=129
maxpower=64
batch_size = int(sys.argv[3])
hidden_delta=256
hidden_note=256
hidden_inst=256
filter_size=128
kernel_size=7 ## midi program changes are by groups
drop_rate=0.2 ## for powerful computer
Normal=120.0
defaultRes=256.0

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
    data=[(0.0,0,0,0)]#tick , note, instrument, power
    instrument = 0 # sets to piano by default
    for track in pattern:
        temp=[(0.0,0,0,0)] #tick, note, instrument, power
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
            elif isinstance(v, midi.NoteOnEvent) and v.data[0]>=21 and v.data[0]<=108 and v.data[1]>0:
                note = v.data[0]-21
                power = v.data[1] / 2
                power = 64 if power>64 else power
                power = 1 if power<1 else power
                power -= 1 ## [0, 63]
                temp.append((accumTick, note, instrument, power))
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
        power= data[i+1][3]
        data[i] = (tick,note,inst,power)
    data = data[0:-2] ## data must have two elements. ow crashs
    return data

def ch_pattern2map(pattern, maxtick): ## tick range [0,63] #64
    ch2ins = dict()
    ResScale = float(pattern.resolution) / float(defaultRes)
    data=[(0.0,0,0,0)]#tick , note, instrument, power
    instrument = 0 # sets to piano by default
    for track in pattern:
        temp=[(0.0,0,0,0)] #tick, note, instrument, power
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
            elif isinstance(v, midi.NoteOnEvent) and v.data[0]>=21 and v.data[0]<=108 and v.data[1]>0:
                ch = v.channel
                note = v.data[0]-21
                power = v.data[1] / 2
                power = 64 if power>64 else power
                power = 1 if power<1 else power
                power -= 1 ## [0, 63]
                temp.append((accumTick, note, ch, power))
        temp = temp[1:-1]
        data.extend(temp)
    data = list(set(data)) ## remove duplicate data
    for i in xrange(len(data)): ## channel -> instrument
        acc = data[i][0]
        note= data[i][1]
        ch  = data[i][2]
        power=data[i][3]
        inst=ch2ins[ch]
        data[i] = (acc,note,inst,power)
    data.sort() ## for better quality
    data = purge(data)
    for i in range(0, len(data)-1):
        tick = data[i+1][0] - data[i][0]
        tick = int(round(tick/ResScale)) ## adjust resolution, downsampling
        tick = maxtick if tick>maxtick else tick ## set a threshold
        note = data[i+1][1]
        inst = data[i+1][2]
        power= data[i+1][3]
        data[i] = (tick,note,inst,power)
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
    powers= np.zeros((len(segment), segLen, 1), dtype=np.uint8)
    insts = np.zeros((len(segment), segLen, maxinst), dtype=np.bool)

    notes_n = np.zeros((len(segment), vecLen), dtype=np.bool)
    times_n = np.zeros((len(segment), maxdelta), dtype=np.bool)
    powers_n= np.zeros((len(segment), 1), dtype=np.uint8)
    insts_n = np.zeros((len(segment), maxinst), dtype=np.bool)
    for i, seg in enumerate(segment):
        for t, note in enumerate(seg):
            times[i, t, int(note[0])] = 1
            notes[i, t, int(note[1])] = 1
            insts[i, t, int(note[2])] = 1
            powers[i, t,0] = int(note[3])
        times_n[i, int(nextseg[i][0])] = 1
        notes_n[i, int(nextseg[i][1])] = 1
        insts_n[i, int(nextseg[i][2])] = 1
        powers_n[i,0] = int(nextseg[i][3])
    return notes, times, insts, powers, notes_n, times_n, insts_n, powers_n

def generator(path_name, step_size, batch_size):
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
                note, time, inst, power, n_note, n_time, n_inst, n_power = seg2vec(seg, nextseg, segLen, vecLen, maxdelta, maxinst)
                seg = None
                nextseg = None
            except:
                pattern = data = seg = nextseg = None
                continue
            for i in xrange(0, len(note)-batch_size, batch_size):
                idx = range(i, i+batch_size)
                yield ([note[idx],time[idx],inst[idx]], [n_note[idx],n_time[idx],n_inst[idx],n_power[idx]])
            l = len(note)%batch_size
            if l > 0:
                idx = range(len(note)-l,len(note))
                yield ([note[idx],time[idx],inst[idx]], [n_note[idx],n_time[idx],n_inst[idx],n_power[idx]])

def main():
    # network:
    # build the model: stacked GRUs
    print('Build model...')
    # network:
    with tf.device('/gpu:0'):
        noteInput  = Input(shape=(segLen, vecLen))
        noteAcc    = SoftAttentionBlock(noteInput, segLen)
        noteEncode = BatchNormalization()(noteAcc)
        noteEncode = GRU(hidden_note, return_sequences=True, dropout=drop_rate)(noteEncode)
        noteEncode = BatchNormalization()(noteEncode)
        noteEncode = GRU(hidden_note, return_sequences=True, dropout=drop_rate)(noteEncode)
        noteEncode = BatchNormalization()(noteEncode)
        noteEncode = GRU(hidden_note, return_sequences=True, dropout=drop_rate)(noteEncode)
        noteEncode = Attention()(noteEncode)

    with tf.device('/gpu:1'):
        deltaInput = Input(shape=(segLen, maxdelta))
        deltaAcc   = SoftAttentionBlock(deltaInput, segLen)
        deltaEncode = BatchNormalization()(deltaAcc)
        deltaEncode = GRU(hidden_delta, return_sequences=True, dropout=drop_rate)(deltaEncode)
        deltaEncode = BatchNormalization()(deltaEncode)
        deltaEncode = GRU(hidden_delta, return_sequences=True, dropout=drop_rate)(deltaEncode)
        deltaEncode = BatchNormalization()(deltaEncode)
        deltaEncode = GRU(hidden_delta, return_sequences=True, dropout=drop_rate)(deltaEncode)
        deltaEncode = Attention()(deltaEncode)

    with tf.device('/gpu:2'):
        instInput = Input(shape=(segLen, maxinst))
        instEncode   = Conv1D(filters=filter_size, kernel_size=kernel_size, padding='same', input_shape=(segLen, maxinst), activation = 'relu')(instInput)
        instEncode   = SoftAttentionBlock(instEncode, segLen)
        instEncode   = BatchNormalization()(instEncode)
        instEncode   = GRU(hidden_inst, return_sequences=True, dropout=drop_rate)(instEncode)
        instEncode   = BatchNormalization()(instEncode)
        instEncode   = GRU(hidden_inst, return_sequences=True, dropout=drop_rate)(instEncode)
        instEncode   = BatchNormalization()(instEncode)
        instEncode   = GRU(hidden_inst, return_sequences=True, dropout=drop_rate)(instEncode)
        instEncode   = Attention()(instEncode)

    with tf.device('/gpu:3'):
        codec = concatenate([noteEncode, deltaEncode, instEncode], axis=-1) ## return last state

        batchNormNote_0 = BatchNormalization()(codec)
        fc_notes = Dense(vecLen*2, kernel_initializer='normal', activation='relu')(batchNormNote_0)
        fc_notes = BatchNormalization()(fc_notes)
        pred_notes = Dense(vecLen, kernel_initializer='normal', activation='softmax', name='note_output')(fc_notes) ## output PMF

        batchNormDelta_0 = BatchNormalization()(codec)
        fc_delta = Dense(maxdelta*2, kernel_initializer='normal', activation='relu')(batchNormDelta_0)
        fc_delta = BatchNormalization()(fc_delta)
        pred_delta = Dense(maxdelta, kernel_initializer='normal', activation='softmax', name='time_output')(fc_delta) ## output PMF

        batchNormInst_0 = BatchNormalization()(codec)
        fc_inst = Dense(maxinst*2, kernel_initializer='normal', activation='relu')(batchNormInst_0)
        fc_inst = BatchNormalization()(fc_inst)
        pred_inst = Dense(maxinst, kernel_initializer='normal', activation='softmax', name='inst_output')(fc_inst) ## output PMF

        batchNormPower_0 = BatchNormalization()(codec)
        fc_power = Dense(maxpower, kernel_initializer='normal', activation='relu')(batchNormPower_0)
        fc_power = BatchNormalization()(fc_power)
        pred_power = Dense(1, kernel_initializer='normal', activation='relu', name='power_output')(fc_power) ## output regression >= 0
    aiComposer = Model([noteInput, deltaInput, instInput], [pred_notes, pred_delta, pred_inst, pred_power])
    checkPoint = ModelCheckpoint(filepath="weights-{epoch:04d}-{loss:.2f}-{val_loss:.2f}.h5", verbose=1, save_best_only=False, save_weights_only=True, period=3)
    Logs = CSVLogger('logs.csv', separator=',', append=True)
    optimizer = RMSprop(lr=learning_rate, decay=learning_rate/epochs, clipnorm=1.)
    if ( os.path.isfile('./top_weight.h5')):  ## fine-tuning
        aiComposer.load_weights('./top_weight.h5')
    aiComposer.summary()
    aiComposer.compile(
            loss = {
                     'note_output':'categorical_crossentropy',
                     'time_output':'categorical_crossentropy',
                     'inst_output':'categorical_crossentropy',
                     'power_output':'mean_squared_error'
                   },
            loss_weights = {
                             'note_output':1,
                             'time_output':1e-1,
                             'inst_output':1e-2,
                             'power_output':1e-3
                           },
            optimizer=optimizer, metrics=['accuracy'])
    aiComposer.fit_generator(generator(str(sys.argv[1]), step_size, batch_size), steps_per_epoch=samples_per_epoch, epochs=epochs, validation_data=generator(str(sys.argv[2]), step_size, batch_size), validation_steps=10, callbacks=[checkPoint, Logs])
    aiComposer.save('./multi.h5')

if __name__ == "__main__":
    main()
