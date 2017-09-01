from __future__ import print_function
import os
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, Flatten, Conv1D
from keras.layers import LSTM, BatchNormalization, RepeatVector, TimeDistributed
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
epochs = int(sys.argv[1])
segLen=128
vecLen=88
maxdelta=128
maxinst=129
maxpower=64
batch_size=128
hidden_delta=256
hidden_note=256
hidden_inst=256
filter_size=128
kernel_size=7 ## midi program changes are by groups
drop_rate=0.2 ## for powerful computer

K.set_floatx(compute_precision);

# build the model: stacked LSTMs
print('Build model...')
with h5py.File('./DataSet.h5','r') as f:
    with h5py.File('./TestSet.h5','r') as testSet:
        vecs0 = f['Note']
        vecs1 = f['Delta']
        vecs2 = f['Inst']
        lab0 = f['NextNote']
        lab1 = f['NextDelta']
        lab2 = f['NextInst']
        lab3 = f['NextPower']

        test_x0 = testSet['Note']
        test_x1 = testSet['Delta']
        test_x2 = testSet['Inst']
        test_y0 = testSet['NextNote']
        test_y1 = testSet['NextDelta']
        test_y2 = testSet['NextInst']
        test_y3 = testSet['NextPower']

        # network:
        with tf.device('/gpu:0'):
            noteInput  = Input(shape=(segLen, vecLen))
            noteEncode = LSTM(hidden_note, input_shape=(segLen, vecLen), return_sequences=True, dropout=drop_rate)(noteInput)
            noteEncode = LSTM(hidden_note, return_sequences=False, dropout=drop_rate)(noteEncode)

        with tf.device('/gpu:1'):
            deltaInput = Input(shape=(segLen, maxdelta))
            deltaEncode = LSTM(hidden_delta, input_shape=(segLen, maxdelta), return_sequences=True, dropout=drop_rate)(deltaInput)
            deltaEncode = LSTM(hidden_delta, return_sequences=False, dropout=drop_rate)(deltaEncode)

        with tf.device('/gpu:2'):
            instInput = Input(shape=(segLen, maxinst))
            instEncode   = Conv1D(filters=filter_size, kernel_size=kernel_size, padding='same', input_shape=(segLen, maxinst), activation = 'relu')(instInput)
            instEncode   = LSTM(hidden_inst, return_sequences=True, dropout=drop_rate)(instEncode)
            instEncode   = LSTM(hidden_inst, return_sequences=False, dropout=drop_rate)(instEncode)

        with tf.device('/gpu:3'):
            codec = concatenate([noteEncode, deltaEncode, instEncode], axis=-1)

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
        earlyStop = EarlyStopping(monitor='loss', patience=50, verbose=1, mode='auto')
        checkPoint = ModelCheckpoint(filepath="./top_weight.h5", verbose=1, save_best_only=True, monitor='loss', mode='min', save_weights_only=True, period=1)
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
        aiComposer.fit( [vecs0, vecs1, vecs2], [lab0, lab1, lab2, lab3], batch_size=batch_size, shuffle=False, epochs=epochs, callbacks=[earlyStop, checkPoint, Logs])
        acc = aiComposer.evaluate([test_x0, test_x1, test_x2], [test_y0, test_y1, test_y2, test_y3], batch_size=batch_size)
        print(acc)
        aiComposer.save('./multi.h5')
