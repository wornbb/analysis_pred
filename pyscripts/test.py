import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,merge, Dropout,Add, LSTM, Bidirectional, BatchNormalization, Input
from keras.datasets import imdb
from keras.utils import to_categorical
import pickle
import tensorflow as tf
from loading import read_violation
from keras import regularizers
from keras.callbacks import ModelCheckpoint, CSVLogger
import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"

[x_train,y_train,x_test,y_test] = pickle.load(open("all_vios.p","rb"))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
x_train = scaler.fit_transform(x_train[:,:-5,0])
x_train = np.expand_dims(x_train, axis=2)
x_test = scaler.fit_transform(x_test[:,:-5,0])
x_test = np.expand_dims(x_test, axis=2)

csv_logger = CSVLogger('training.csv',append=True)
rnn_dropout = 0.1
for m in [32,48,64]:
    for n in range(34,36,4):
        inputs = Input(shape=(34,1))
        for rnn in range(5):
            if rnn == 0:
                lstm = LSTM(n, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=True)(inputs)
                node = Add()([inputs, lstm])
            elif rnn < 4:
                lstm = LSTM(n, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=True)(node)
                node = Add()([node, lstm])
            else:
                node = LSTM(n,recurrent_dropout=rnn_dropout, dropout=rnn_dropout)(node)
                #node = Add()([node, lstm])

        #model.add(Dense(m, activation='selu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        #model.add(BatchNormalization())
        #model.add(Dense(2, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros'))
        prediction = Dense(2, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(node)
        model = keras.models.Model(inputs=inputs, outputs=prediction)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        model.summary()
        print('Train...')
        filepath = "nn." + str(m) + ".biLSTM." + str(n) + ".{epoch:02d}-{val_loss:.3f}.hdf5"
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

