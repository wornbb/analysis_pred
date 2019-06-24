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
def preprocess(batch, base_volt, threshold):
    """Find the correct Y given x (batch)
    
    Arguments:
        batch {np 2D} -- loaded voltage on PDN grid
        base_volt {int } -- reference normal voltage
        threshold {int} -- threshold to define violation (in percentage)
    """
    batch = batch / base_volt
    overshoot = batch >= (1 + threshold/100)
    droop = batch <= (1 - threshold/100)
    labeled = overshoot + droop
    batch = (batch - 0.9 * base_volt) * 10

    return batch.astype('float32') , labeled.astype('int')
def prepare_xy(x, labeled, timestep):
    if x.shape[1] % timestep:
        print(x.shape[1] % timestep)
        print('sample/timestep mismatch')
        return 0
    # parameter
    timestep_per_cycle = 5
    input_cycle = 0
    predi_cycle = 0
    inputStep = timestep_per_cycle * input_cycle
    prediStep = timestep_per_cycle * predi_cycle 

    # prepare loop
    startP  = inputStep + prediStep + 1
    half_sample = np.sum(np.any(labeled,0))
    sample_size = 2 * half_sample
    sensor_num = 1
    dim = int(np.sqrt(x.shape[0]))
    X = np.zeros((sample_size, dim, dim, 1))
    Y = np.zeros((sample_size, sensor_num))
    sample_counter = -1
    balance_counter = 0
    for col in range(startP, x.shape[1]):
        if np.any(labeled[:, col]) == 1:
            balance_counter += 1
            sample_counter += 1
            X[sample_counter, :, :, 0] = x[:, col - prediStep].reshape(dim, dim)
            Y[sample_counter, 0] = 1
        else:
            if balance_counter > 0:
                balance_counter -= 1
                sample_counter += 1
                X[sample_counter, :, :, 0] = x[:, col - prediStep].reshape(dim, dim)
                Y[sample_counter, 0] = 0
    print("balance_ounter = ", balance_counter)                 
    return X.astype('float32'), Y.astype('int')

  
with open('saved_data.pk', 'rb') as f:
  save = pickle.load(f)
  
base_volt = 1
threshold = 3
timestep = 50

x_train = save[0]
x_test = save[1]
print(x_train.shape)
x_train, y_train = preprocess(x_train, base_volt, threshold)
x_train, y_train = prepare_xy(x_train, y_train, timestep)
print(y_train.shape)
print(x_train.shape)
x_test, y_test = preprocess(x_test, base_volt, threshold)
x_test, y_test = prepare_xy(x_test, y_test, timestep)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



model = Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=4, strides=4, activation='relu', input_shape=(76, 76, 1)))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'))
model.add(layers.MaxPool2D())
#model.add(layers.Conv2D(filters=128, kernel_size=3, strides=3, activation='relu'))
#model.add(layers.MaxPool2D())
model.add(layers.Flatten())
#model.add(LSTM(10, kernel_initializer='random_uniform', bias_initializer='zeros'))
#model.add(Bidirectional(LSTM(128, input_shape=(timestep,9), kernel_initializer='random_uniform', bias_initializer='zeros')))
model.add(Dropout(0.2))
#model.add(layers.Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(2, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['binary_accuracy', 'categorical_accuracy'])
#model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])


print('Train...')
batch_size = 15

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=300,
          verbose=1)
scores = model.evaluate(x_test, y_test, verbose=0)
#print(sum(y_test)/len(y_train))
from sklearn.metrics import confusion_matrix
result = model.predict_on_batch(x_test)
#confusion_matrix(y_test, result>0.5)
print("Accuracy: %.2f%%" % (scores[1]*100))