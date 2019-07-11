import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.datasets import imdb
from keras.utils import to_categorical
import pickle
import tensorflow as tf
from loading import read_violation
from keras import regularizers
from keras.callbacks import ModelCheckpoint, CSVLogger
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"

[x_train,y_train,x_test,y_test] = pickle.load(open("all_vios.p","rb"))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
x_train = scaler.fit_transform(x_train[:,:,0])
x_train = np.expand_dims(x_train, axis=2)
x_test = scaler.fit_transform(x_test[:,:,0])
x_test = np.expand_dims(x_test, axis=2)

csv_logger = CSVLogger('training.log',append=True)

for n in range(20,50):
    model = Sequential()
    print(len(y_test), sum(y_test))
    model.add(LSTM(n, kernel_initializer='random_uniform', bias_initializer='zeros', return_sequences=True))
    model.add(LSTM(n, kernel_initializer='random_uniform', bias_initializer='zeros', return_sequences=True))
    model.add(LSTM(n, kernel_initializer='random_uniform', bias_initializer='zeros', return_sequences=True))
    model.add(LSTM(n, kernel_initializer='random_uniform', bias_initializer='zeros'))
    #model.add(Bidirectional(LSTM(128, input_shape=(timestep,9), kernel_initializer='random_uniform', bias_initializer='zeros')))
    model.add(Dense(32, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dropout(0.15))
    model.add(Dense(16, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dropout(0.15))
    model.add(Dense(2, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    print('Train...')
    filepath = "model.{epoch:02d}-{val_loss:.3f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy',save_best_only=True, verbose=1, mode='max')
    batch_size = 5
    model.fit(x_train[::100,:,:], y_train[::100],
            batch_size=batch_size,
            validation_data=(x_test[::1000,:,:],y_test[::1000]),
            epochs=25,
            callbacks=[checkpoint, csv_logger],
            verbose=1)
# model.fit(x_train[::100,:,:], y_train[::100],
#           batch_size=batch_size,
#           validation_data=(x_test[::1000,:,:],y_test[::1000]),
#           epochs=10,
#           verbose=1)
#pickle.dump( model, open( "single_sensor_lstm10.p", "wb" ) )


# scores = model.evaluate(x_test, y_test, verbose=0)
# print(sum(y_test)/len(y_train))
# print("Accuracy: %.2f%%" % (scores[1]*100))

