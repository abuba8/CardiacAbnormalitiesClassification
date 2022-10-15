from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Add
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D, ZeroPadding1D
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU


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
    
    X = Conv1D(filters = 64, kernel_size = 7, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPool1D(pool_size=3, strides=2, padding='same')(X)
    
    X = Conv1D(filters = 64, kernel_size = 1, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = inception_block(X)
    X = inception_block(X)
    X = inception_block(X)
    X = Dropout(0.2)(X)

    
    # # split for attention
    # attention_data = keras.layers.Lambda(lambda x: x[:,:,:256])(X)
    # attention_softmax = keras.layers.Lambda(lambda x: x[:,:,256:])(X)
    # # attention mechanism
    # attention_softmax = keras.layers.Softmax()(attention_softmax)
    # multiply_layer = keras.layers.Multiply()([attention_softmax,attention_data])
    # dense_layer = keras.layers.Dense(units=256,activation='sigmoid')(multiply_layer)
    # X = BatchNormalization()(dense_layer)
    # flatten_layer = keras.layers.Flatten()(dense_layer)
    # output_layer = keras.layers.Dense(units=27,activation='sigmoid')(flatten_layer)

    X = MaxPool1D(pool_size=7, strides=2, padding='same')(X)
    
    X = GlobalAveragePooling1D()(X)
    X = Dense(27,activation='sigmoid')(X)
    
    model = Model(inputs = X_input, outputs = X, name='Inception')
    
    return model
