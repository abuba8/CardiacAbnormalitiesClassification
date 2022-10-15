import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Add
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D, ZeroPadding1D, LSTM, Bidirectional
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.layers.merge import concatenate
from scipy import optimize
from scipy.io import loadmat
import os
from keras.layers import LeakyReLU
import tensorflow_addons as tfa



""" Convolutional, Residual, and Inception block
#### Proposed Architecture:

Firstly, we used 1D convolutional layers to extract high-level features 
with batch normalization, and in the 1D max pool layer the input 
dimensions were reduced. Then used a residual block for extracting 
low-level feature representations and connected with a skip connection 
so that the weights are transferred later in the network so that 
we dont have a vanishing gradient problem. Then used the inception 
network to extract more low-dimensional features parallelly followed 
by 3 convolutional blocks. Afterward, a dropout layer if the network 
overfits then 20 percent of the neurons should be dropped. Then in the last 
step, we used a convolutional block three times to extract complex 
patterns in the data due to high variance. Finally, the 1D max pool 
layer and 1D global average pooling layer were used then we flatten 
the data into a single dimension to connect it by a dense layer to 
classify.

In the first convolutional layer, 512 filters were used with a size 
of 5x5 followed by a batch normalization layer and relu activation. 
1D Max pool layer with a filter size of 3x3 and stride 2x2 was used 
for downsampling the data and preserving the prominent features. 

In residual block, three stacks of a 1D convolutional layer with batch 
normalization layer and leaky relu were used. The alpha for leaky relu 
was again set to 1e-2. The number of filters of each stack of the 
convolutional layer was 128, 128, and 256 respectively with a size of 
1x1. In the skip connection, another convolutional layer with 
256 filters was used with batch normalization which was then connected 
with the output of the third stack of the residual block to preserve 
the weights.

In inception block stacks of 1D convolutional layers followed by 
batch normalization layer and leaky relu with alpha 1e-2 were used. 
The filters used in all stacks were 64, with kernel sizes 1, 3, and 5 
respectively. 

Lastly, three convolutional blocks were added. In the first stack of 
convolutional block 1D convolutional layer was added with 128 filters 
and 5x5 filter size and stride 1x1 followed by an instance normalization
layer and parametric relu as activation function. Then a dropout layer 
was added with a probability of 20 percent and a 1D max pooling layer 
with filter size 2x2 was added. In the second convolutional block, 
only the number of filters in the convolutional layer was different 
than the first stack. In the second stack, 256 filters with 11x11 size 
were used in 1D convolutional layer. Whereas, in the third stack of 
the convolutional block we didnt add the 1D pooling layer and used 
512 filters with size 21x21 in the convolutional layer.

"""

def convolutional_block(X_input):
    X = keras.layers.Conv1D(filters=128,kernel_size=5,strides=1,padding='same')(X_input)
    X = tfa.layers.InstanceNormalization()(X)
    X = keras.layers.PReLU(shared_axes=[1])(X)
    X = keras.layers.Dropout(rate=0.2)(X)
    X = keras.layers.MaxPooling1D(pool_size=2)(X)
    # conv block -2
    X = keras.layers.Conv1D(filters=256,kernel_size=11,strides=1,padding='same')(X)
    X = tfa.layers.InstanceNormalization()(X)
    X = keras.layers.PReLU(shared_axes=[1])(X)
    X = keras.layers.Dropout(rate=0.2)(X)
    X = keras.layers.MaxPooling1D(pool_size=2)(X)
    # conv block -3
    X = keras.layers.Conv1D(filters=512,kernel_size=21,strides=1,padding='same')(X)
    X = tfa.layers.InstanceNormalization()(X)
    X = keras.layers.PReLU(shared_axes=[1])(X)
    X = keras.layers.Dropout(rate=0.2)(X)

    return X


def residual_block(X, f, filters, s = 2):
    F1, F2, F3 = filters
    
    X_shortcut = X

    X = Conv1D(filters=F1, kernel_size=1, strides = s)(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.01)(X)
    
    X = Conv1D(filters=F2, kernel_size=1, strides = s)(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv1D(filters=F3, kernel_size=1, strides = 1)(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.01)(X)

    X_shortcut = Conv1D(filters=F3, kernel_size=1, strides = s)(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)
    
    X = Add()([X,X_shortcut])
    X = LeakyReLU(alpha=0.01)(X)
    
    return X



def inception_block(prev_layer):
    
    conv1=Conv1D(filters = 64, kernel_size = 1, padding = 'same')(prev_layer)
    conv1=BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.01)(conv1)
    # conv1=Activation('relu')(conv1)
    
    conv3=Conv1D(filters = 64, kernel_size = 1, padding = 'same')(prev_layer)
    conv3=BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.01)(conv3)
    conv3=Conv1D(filters = 64, kernel_size = 3, padding = 'same')(conv3)
    conv3=BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.01)(conv3)
    
    conv5=Conv1D(filters = 64, kernel_size = 1, padding = 'same')(prev_layer)
    conv5=BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.01)(conv5)
    conv5=Conv1D(filters = 64, kernel_size = 5, padding = 'same')(conv5)
    conv5=BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.01)(conv5)
    
    pool= MaxPool1D(pool_size=3, strides=1, padding='same')(prev_layer)
    convmax=Conv1D(filters = 64, kernel_size = 1, padding = 'same')(pool)
    convmax=BatchNormalization()(convmax)
    convmax = LeakyReLU(alpha=0.01)(convmax)
    
    layer_out = concatenate([conv1, conv3, conv5, convmax], axis=1)
    
    return layer_out




def inception_model(input_shape):
    X_input=Input(input_shape)
    
    X = ZeroPadding1D(3)(X_input)
    
    X = Conv1D(filters = 512, kernel_size = 5, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPool1D(pool_size=3, strides=2, padding='same')(X)
    
    

    X = residual_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = inception_block(X)
    X = residual_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = inception_block(X)
    X = residual_block(X, f = 3, filters = [64, 64, 256], s = 1)


    X = Dropout(0.2)(X)

    X = convolutional_block(X)

    X = MaxPool1D(pool_size=7, strides=2, padding='same')(X)
    
    X = GlobalAveragePooling1D()(X)
    X = Dense(27,activation='sigmoid')(X)
    
    model = Model(inputs = X_input, outputs = X, name='Inception')
    
    return model