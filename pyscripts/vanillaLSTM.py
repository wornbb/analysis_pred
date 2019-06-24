import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.datasets import imdb
from keras.utils import to_categorical
import pickle
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

with open('data_train.pk', 'rb') as f:
    [x_train, y_train, x_test, y_test] = pickle.load(f)


class_weight = {0: 1,1:1}
model = Sequential()
print(x_train.shape)
model.add(LSTM(10, kernel_initializer='random_uniform', bias_initializer='zeros'))
#model.add(Bidirectional(LSTM(128, input_shape=(timestep,9), kernel_initializer='random_uniform', bias_initializer='zeros')))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
batch_size = 10

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=120,
          verbose=1)
scores = model.evaluate(x_test, y_test, verbose=0)
print(sum(y_test)/len(y_train))
print("Accuracy: %.2f%%" % (scores[1]*100))

