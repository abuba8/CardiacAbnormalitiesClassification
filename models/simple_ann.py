import tensorflow as tf
from keras.layers import Dense
from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential


ann_model = Sequential()
ann_model.add(Dense(50, activation='relu', input_shape=(5000,12)))
ann_model.add(Dense(50, activation='relu'))
ann_model.add(Dense(50, activation='relu'))
ann_model.add(Dense(50, activation='relu'))
ann_model.add(GlobalAveragePooling1D())
ann_model.add(Dense(27, activation='softmax'))


