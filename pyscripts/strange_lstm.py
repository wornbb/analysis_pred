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

f_list = [
"balanced_gird_sensor." + "blackscholes2c" + ".h5",
"balanced_gird_sensor." + "bodytrack2c" + ".h5",
"balanced_gird_sensor." + "freqmine2c"+ ".h5",
"balanced_gird_sensor." + "facesim2c"+ ".h5",
]
with h5py.File(f_list[0], 'r') as f:
      x_shape = f["data"].shape
      x_type = f["data"].dtype
      y_type = f["tag"].dtype

margins = np.array([1 + 4/100, 1 - 4/100]) * 1
with h5py.File("lstm_grid_batch_train.h5", 'w') as hf:
    x = hf.create_dataset('x', shape=(1, x_shape[1], 1), maxshape=(None, x_shape[1], 1))
    y = hf.create_dataset('y', shape=(1,), maxshape=(None,))
    for fname in f_list:
        with h5py.File(fname, 'r') as data:
            girds = data['x'][()]
        for index in range(girds.shape[0]):
            higher = girds[index,:,:,0] > margins[0]
            lower = gird[index,:,:,0] < margins[1]
            vios = np.bitwise_or(higher, lower)
            total_vios = np.sum(vios)
            if total_vios >= 0:
                vio_traces = grid[]
# with h5py.File(save_fname,"r") as hf:
#         x = hf["x"].value
#         y= hf["y"].value
# dirty fixing
x = np.squeeze(x, axis=(0,2))
y = y.flatten().astype('int')
shuffle_index = np.arange(y.shape[0])
np.random.shuffle(shuffle_index)
x = x[shuffle_index,:]
y = y[shuffle_index]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
pred_str = 5
scaled_x = scaler.fit_transform(x[:,:-pred_str])
scaled_x = np.expand_dims(scaled_x, axis=2)
train_size = x.shape[0]//4
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
for s in range(3,6):
    for n in range(5,50,10):
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
                validation_data=(x_test[::400,:,:],np.array(y_test[::400])),
                epochs=15,
                callbacks=callbacks,
                verbose=1)