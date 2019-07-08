 #y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

#class_weight = {0: (len(y_train)-sum(y_test)/len(y_train)),
#                1: sum(y_test)/len(y_train)}
from keras.layers import Conv1D, MaxPooling1D
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Activation, TimeDistributed, Flatten, Conv2D
from keras.datasets import imdb
from keras.utils import to_categorical
from keras import optimizers
from keras import layers
import pickle
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

cnn = Sequential()
cnn.add(layers.Conv1D(filters=32, kernel_size=4, strides=4, activation='relu'))
cnn.add(layers.MaxPool1D())
cnn.add(layers.Conv1D(filters=64, kernel_size=3, strides=2, activation='relu'))
cnn.add(layers.MaxPool1D())
cnn.add(layers.Flatten())

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.datasets import imdb
from keras.utils import to_categorical
import pickle
import tensorflow as tf
from loading import read_violation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"

[x_train,y_train,x_test,y_test] = pickle.load(open("all_vios.p","rb"))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train[:,:-5,0])
x_train = np.expand_dims(x_train, axis=2)
x_test = scaler.fit_transform(x_test[:,:-5,0])
x_test = np.expand_dims(x_test, axis=2)

class_weight = {0: 1,1:1}
lstm = Sequential()
print(len(y_test), sum(y_test))
lstm.add(LSTM(84, kernel_initializer='random_uniform', bias_initializer='zeros'))

model = Sequential()
model.add(layers.concatenate([lstm, cnn.output]))
model.add(Flatten())
model.add(Dense(2, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
batch_size = 1000

model.fit([x_train,x_train], y_train,
          batch_size=batch_size,
          validation_data=([x_test,x_test],y_test),
          epochs=20,
          verbose=1)
#pickle.dump( model, open( "single_sensor_lstm10.p", "wb" ) )


# scores = model.evaluate(x_test, y_test, verbose=0)
# print(sum(y_test)/len(y_train))
# print("Accuracy: %.2f%%" % (scores[1]*100))



