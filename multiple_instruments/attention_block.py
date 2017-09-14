import keras
from keras.layers import Dense, Permute, merge, Input, BatchNormalization
from keras.layers.merge import Multiply
import keras.backend as K

def SoftAttentionBlock(inputs, input_ts=None):
    ## input: (batch_size, time_step, features)
    if input_ts is None:
        input_ts = int(inputs.shape[1])
    x = Permute((2,1))(inputs) ## (batch_size, features, time_step)
    x = BatchNormalization()(x)
    x = Dense(input_ts, activation='softmax')(x)
    x = Permute((2,1))(x)
    attention = Multiply()([x, inputs])
    return attention

