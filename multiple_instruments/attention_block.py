import keras
from keras.layers import Dense, Permute, merge, Input, BatchNormalization, Lambda, RepeatVector, Reshape
from keras.layers.merge import Multiply
from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer
import keras.backend as K

'''
The idea of SoftAttentionBlock are from:
https://github.com/philipperemy/keras-attention-mechanism
'''

def SoftAttentionBlock(inputs, input_ts=None, input_dim=None, trainable=True):
    ## input: (batch_size, time_step, features)
    if input_ts is None:
        input_ts = int(inputs.shape[1])
    if input_dim is None:
        input_dim = int(inputs.shape[2])
    x = Permute((2,1))(inputs) ## (batch_size, features, time_step)
    x = Reshape((input_dim, input_ts))(x)
    x = BatchNormalization(trainable=trainable)(x)
    x = Dense(input_ts, activation='softmax', trainable=trainable)(x)
    x = Permute((2,1))(x)
    attention = Multiply()([x, inputs])
    return attention
