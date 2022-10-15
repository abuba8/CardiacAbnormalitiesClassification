import tensorflow as tf
from keras.layers import Dense, Activation, BatchNormalization
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D
from keras.models import Sequential


lenet_5_model=Sequential()

lenet_5_model.add(Conv1D(filters=6, kernel_size=3, padding='same', input_shape=(5000,12)))
lenet_5_model.add(BatchNormalization())
lenet_5_model.add(Activation('relu'))
lenet_5_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

lenet_5_model.add(Conv1D(filters=16, strides=1, kernel_size=5))
lenet_5_model.add(BatchNormalization())
lenet_5_model.add(Activation('relu'))
lenet_5_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))

lenet_5_model.add(GlobalAveragePooling1D())
lenet_5_model.add(Dense(64, activation='relu'))
lenet_5_model.add(Dense(32, activation='relu'))
lenet_5_model.add(Dense(27, activation = 'softmax'))

