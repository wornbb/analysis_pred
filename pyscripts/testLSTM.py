import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.datasets import imdb
from keras.utils import to_categorical
import pickle
import tensorflow as tf
from loading import read_violation
from keras import regularizers
from keras.callbacks import ModelCheckpoint, CSVLogger
import keras as keras
from clr_callback import*
import h5py

fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"
save_fname = "combined_lstm_training.data"
with h5py.File(save_fname,"r") as hf:
        x = hf["x"].value
        y= hf["y"].value
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
pred_str = 5
scaled_x = scaler.fit_transform(x[:,:-pred_str,0])
test_size = 1000
x_train = scaled_x[:-test_size,:]
x_test = scaled_x[-test_size:,:]
y_train = y[:]
print(x_train.shape)
csv_logger = CSVLogger('training.csv',append=True)
from keras.layers import Dense,merge, Dropout,Add, LSTM, Bidirectional, BatchNormalization, Input, Permute
for n in range(25,26):
    for m in  range(25,226):
        model = Sequential()
        print(len(y_test), sum(y_test))
        model.add(Bidirectional(LSTM(n, kernel_initializer='random_uniform', bias_initializer='zeros', return_sequences=True)))
        model.add(Bidirectional(LSTM(n, kernel_initializer='random_uniform', bias_initializer='zeros', return_sequences=True)))
        model.add(Bidirectional(LSTM(n, kernel_initializer='random_uniform', bias_initializer='zeros', return_sequences=True)))
        model.add(Bidirectional(LSTM(n, kernel_initializer='random_uniform', bias_initializer='zeros')))
        selu_ini = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/34, seed=None)

        model.add(Dense(m, activation='selu', kernel_initializer=selu_ini, bias_initializer='zeros'))
        model.add(Dropout(0.15))
        model.add(Dense(m//2, activation='selu', kernel_initializer=selu_ini, bias_initializer='zeros'))
        model.add(Dropout(0.15))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print('Train...')
        filepath = "lstm." + str(n) + ".nn." + str(m) + ".{epoch:02d}-{val_acc:.3f}-{val_loss:.3f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc',save_best_only=True, verbose=1, mode='max')
        batch_size = 5
        model.fit(x_train[::100,:,:],np.array(y_train[::100]),
                batch_size=batch_size,
                validation_data=(x_test[::1000,:,:],np.array(y_test[::1000])),
                epochs=18,
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

