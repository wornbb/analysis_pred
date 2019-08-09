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
import os
fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"

save_fname = "lstm_2c.h5"
with h5py.File(save_fname,"r") as hf:
        x = hf["x"][()]
        y= hf["y"][()]
# dirty fixing
x = np.squeeze(x, axis=(2))
y = y.flatten().astype('int')
shuffle_index = np.arange(y.shape[0])
np.random.shuffle(shuffle_index)
x = x[shuffle_index,:]
y = y[shuffle_index]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
pred_str = 5
scaled_x = scaler.fit_transform(x[:,:])
scaled_x = np.expand_dims(scaled_x, axis=2)
train_size = int(x.shape[0]//8)
x_train = scaled_x[:train_size,:]
x_test = scaled_x[train_size:,:]

y_train = y[:train_size]
y_test = y[train_size:]

log_file = 'training_residual.csv'
try:
    os.remove(log_file)
except:
    pass
csv_logger = CSVLogger(log_file, append=True)
rnn_dropout = 0.4
from keras.layers import Dense,merge, Dropout,Add, LSTM, Bidirectional, BatchNormalization, Input, Permute
m = 32
for s in range(4,5):
    for n in range(60,61,10):
        inputs = Input(shape=(34,1))
        #shuffled = Permute((2,1), input_shape=(34,1))(inputs)
        for rnn in range(s):
            if rnn == 0:
                lstm = Bidirectional(LSTM(n, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=True))(inputs)
                node = Add()([inputs, lstm])
            elif rnn < s - 1:
                lstm = Bidirectional(LSTM(n, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=True))(node)
                node = Add()([node, lstm])
            else:
                node = Bidirectional(LSTM(n,recurrent_dropout=rnn_dropout, dropout=rnn_dropout,))(node)
                #node = Add()([node, lstm])
        selu_ini = keras.initializers.RandomNormal(mean=0.0, stddev=1/40, seed=None)
        node = Dense(m, activation='selu', kernel_initializer=selu_ini)(node)
        node = BatchNormalization()(node)
        # node = Dense(m, activation='selu', kernel_initializer=selu_ini)(node)
        # node = BatchNormalization()(node)
        prediction = Dense(1, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros')(node)
        model = keras.models.Model(inputs=inputs, outputs=prediction)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.summary()
        print('Train...')
        filepath = "residual." + str(s) + ".biLSTM." + str(n) + ".{epoch:02d}-{val_acc:.3f}-{val_loss:.3f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc',save_best_only=True, verbose=1, mode='max')
        clr = CyclicLR(base_lr=0.05, max_lr=0.15, mode='triangular2')
        batch_size = 64
        callbacks = [checkpoint, csv_logger]
        model.fit(x_train, np.array(y_train),
                batch_size=batch_size,
                validation_data=(x_test[::401,:,:],np.array(y_test[::401])),
                epochs=50,
                callbacks=callbacks,
                verbose=1)