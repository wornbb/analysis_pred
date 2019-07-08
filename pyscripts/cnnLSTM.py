 #y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

#class_weight = {0: (len(y_train)-sum(y_test)/len(y_train)),
#                1: sum(y_test)/len(y_train)}
from keras.layers import Conv1D, MaxPooling1D
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Activation
from keras.datasets import imdb
from keras.utils import to_categorical
from keras import optimizers
import pickle
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

with open('data_train.pk', 'rb') as f:
    [x_train, y_train, x_test, y_test] = pickle.load(f)



model = Sequential()
print(x_train.shape)
model.add(Conv1D(filters=64, kernel_size=3, strides=1))
model.add(Activation('relu'))
model.add(MaxPooling1D())
#model.add(LSTM(10, kernel_initializer='random_uniform', bias_initializer='zeros'))
#model.add(Bidirectional(LSTM(128, input_shape=(timestep,9), kernel_initializer='random_uniform', bias_initializer='zeros')))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))

model.add(Dense(1, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
#model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])


print('Train...')
batch_size = 10

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=30,
          verbose=1)
#scores = model.evaluate(x_test, y_test, verbose=0)
#print(sum(y_test)/len(y_train))
print(model.predict(x_test)[:20])
#print("Accuracy: %.2f%%" % (scores[1]*100))