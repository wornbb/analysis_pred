import tensorflow as tf
import numpy as np
from loading import generate_prediction_data
from clr_callback import*
import tensorflow
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import TimeDistributed, Input, Dense, BatchNormalization, Flatten
from tensorflow.keras.layers import Dense, Dropout,Add, LSTM, Bidirectional, BatchNormalization, Input, Permute
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from keras.utils.io_utils import HDF5Matrix
from preprocessing import lstm_sweep
from sklearn.metrics import make_scorer, mean_squared_error
import pickle
from loading import *
from clr_callback import*
import h5py
import os
class voltnet_model():
    def __init__(self, pred_str=0):
        self.pred_str = pred_str
        self.LSTM = voltnet_LSTM(pred_str=self.pred_str)
        self.callbacks = []
        self.prepare_callback_prune()
        self.train_size = 10000
        self.test_size = 3000
    def load_LSTM(self, model_name='residual.3.biLSTM.45.15-0.997-0.008.hdf5'):
        self.LSTM = load_frozen_lstm(model_name)
    def load_h5(self, fname, categorical=True):

        with h5py.File(fname,'r') as f:
            tag = f["y"][:self.train_size + self.test_size]
            x = f["x"][:self.train_size + self.test_size,...]
        if categorical:
            tag = to_categorical(tag[:self.train_size + self.test_size])
        
        #self.x_train = HDF5Matrix(datapath=fname, dataset='x', start=0, end=train_size)
        #self.x_test = HDF5Matrix(datapath=fname, dataset='x', start=train_size, end=train_size+test_size)
        return [x, tag]

        # self.y_train = tag[:train_size,:]
        # self.y_test = tag[train_size:train_size+test_size,:]
    def divide_data(self, data):
        train = data[:self.train_size,...]
        test = data[self.train_size : self.train_size + self.test_size,...]
        return [train, test]
    def fit(self, lstm_train_data="Scaled_lstm_2c.h5.0.25", gird_train_data="Scaled_VoltNet_2c.h5"):
        [x, y] = self.load_h5(fname=lstm_train_data, categorical=False)
        [x_train, x_test] = self.divide_data(x)
        [y_train, y_test] = self.divide_data(y)
        self.LSTM.fit(x_train, y_train, x_test, y_test)
        [x, y] = self.load_h5(fname=gird_train_data)
        sweeper = lstm_sweep(scaled_grid_fname=gird_train_data, save_fname='F:\\lstm_data\\prob_distribution.h5')
        sweeper.lstm_model = self.LSTM
        sweeper.process()
        print("Sweeping completed")
        self.fit_from_prob()
    def fit_from_prob(self, fname='F:\\lstm_data\\prob_distribution.h5'):
        [x, y] = self.load_h5(fname=fname)
        [x_train, x_test] = self.divide_data(x)
        [y_train, y_test] = self.divide_data(y)
        self.fit_selector_(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        self.fit_predictor_(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    def fit_selector_(self, x_train, y_train, x_test, y_test):
        inputs = Input(shape=(x_train.shape[1],))
        outputs = sparsity.prune_low_magnitude(Dense(2, activation='softmax'), **self.pruning_params)(inputs)
        self.selector = keras.models.Model(inputs=inputs, outputs=outputs)
        self.selector.summary()
        self.selector.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        self.selector.fit(x_train, y_train,
            validation_data=(x_test, y_test),
            batch_size=32,
            epochs=15,      
            shuffle='batch',
            callbacks=self.callbacks,
            verbose=1)
    def fit_predictor_(self, x_train, y_train, x_test, y_test):
        output_weights = self.selector.layers[-1].get_weights()[0]
        weight_norm = np.linalg.norm(output_weights, axis=1)
        self.selected_sensors = weight_norm > 0.3
        print(np.sum(self.selected_sensors))
        x_train = x_train[:,self.selected_sensors]
        x_test = x_test[:, self.selected_sensors]
        n = 64
        inputs = Input(shape=(x_train.shape[1]))
        selu_ini = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/34, seed=None)
        outputs = Dense(n, activation='selu', kernel_initializer=selu_ini, bias_initializer='zeros')(inputs)
        outputs = BatchNormalization()(outputs)
        # outputs = Dense(n/4, activation='selu', kernel_initializer=selu_ini, bias_initializer='zeros')(outputs)
        # outputs = BatchNormalization()(outputs)
        outputs = Dense(2, activation='softmax')(inputs)
        self.predictor = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        self.predictor.summary()
        ck_point = "pruned.nn."+ str(n) + ".{epoch:02d}-{val_categorical_accuracy:.3f}-{val_loss:.3f}.hdf5"
        self.predictor.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        self.predictor.fit(x_train, y_train,
            shuffle='batch',
            validation_data=(x_test[:,:],y_test[:,:]),
            batch_size=1,
            epochs=15,      
            callbacks=[ CSVLogger('pruned_training.csv',append=True),
                        ModelCheckpoint(ck_point, monitor='val_categorical_accuracy',save_best_only=True, verbose=1, mode='max'),
                        ],
            verbose=1)
        eva=self.predictor.evaluate(x=x_test,y=y_test)
        print(eva)
    def save(self):
        self.predictor.save('voltnet.predictor.h5')
        pickle.dump(self.selected_sensors, open("voltnet.selected.pk",'wb'))
    def load(self, predictor_fname='voltnet.predictor.h5', selection_fname="voltnet.selected.pk"):
        self.predictor = tf.keras.models.load_model(predictor_fname)
        self.selected_sensors = pickle.load(open(selection_fname,'rb'))
    def prepare_callback_prune(self):
        self.pruning_params = {
            'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                        final_sparsity=0.98,
                                                        begin_step=500,
                                                        end_step=1500,
                                                        frequency=100)
        }
        logdir = 'prune.log'
        self.updater = sparsity.UpdatePruningStep()
        self.summary = sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
        self.callbacks.append(self.updater)
        self.callbacks.append(self.summary)
    def prepare_callback_ckp(self):
        filepath = "Voltnet.Complete" + ".{epoch:02d}-{val_categorical_accuracy:.3f}-{val_loss:.3f}.hdf5"
        self.ckp = ModelCheckpoint(filepath, monitor='val_categorical_accuracy',save_best_only=True, verbose=1, mode='max')
        self.callbacks.append(self.ckp)
    def prepare_callback_csv(self):
        self.csv_logger = CSVLogger('pruned_training.csv',append=True)
        self.callbacks.append(self.csv_logger)
    def predict(self, x):
        result = self.predictor.predict(x[:,self.selected_sensors])
        #print(result)
        return int(np.argmax(result, axis=1))
    def evaluate(self, x, y):
        X = x[:,self.selected_sensors]
        y_pred = self.predictor.predict(X)
        return [0, mean_squared_error(y, y_pred)]
class voltnet_LSTM():
    def __init__(self, pred_str=0):
        self.pred_str = pred_str
        self.callbacks = []
        self.prepare_callback_ckp()
        self.prepare_callback_csv()
    def fit(self, x, y, x_test, y_test):
        # NN hyperparameter
        rnn_dropout = 0.4
        m = 32
        s = 3
        n = 45
        batch_size = 128
        # setting up lstm
        inputs = Input(shape=(34,1))
        for rnn in range(s):
            if rnn == 0:
                lstm = Bidirectional(LSTM(n, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=True))(inputs)
                node = Add()([inputs, lstm])
            elif rnn < s - 1:
                lstm = Bidirectional(LSTM(n, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=True))(node)
                node = Add()([node, lstm])
            else:
                node = Bidirectional(LSTM(n,recurrent_dropout=rnn_dropout, dropout=rnn_dropout,))(node)
        selu_ini = tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=1/40, seed=None)
        node = Dense(m, activation='selu', kernel_initializer=selu_ini)(node)
        node = BatchNormalization()(node)
        outputs = Dense(1, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros')(node)
        self.model = tensorflow.keras.models.Model(inputs=inputs, outputs=outputs)
        #rmsprop = tensorflow.keras.optimizers.RMSprop(lr=0.005)
        self.model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
        self.model.summary()
        print('Train...')  
        self.model.fit(x[:x.shape[0]-self.pred_str,...], y[self.pred_str:],
            validation_data=(x_test, y_test),
            batch_size=batch_size,
            shuffle="batch",
            epochs=50,
            callbacks=self.callbacks,
            verbose=1)
    def prepare_callback_clr(self):
        self.clr = CyclicLR(base_lr=0.05, max_lr=0.15, mode='triangular2')
        self.callbacks.append(self.clr)
    def prepare_callback_ckp(self):
        filepath = "voltnet." + ".selector."  +"pred_str." + str(self.pred_str) + ".{epoch:02d}-{val_acc:.3f}-{val_loss:.3f}.hdf5"
        self.checkpoint = ModelCheckpoint(filepath, monitor='val_acc',save_best_only=True, verbose=1, mode='max')
        self.callbacks.append(self.checkpoint)
    def prepare_callback_csv(self):
        fname = "lstm.pred_str." + str(self.pred_str) + ".log"
        try:
            os.remove(fname)
        except:
            pass
        self.csv_logger = CSVLogger(fname, append=True)
        self.callbacks.append(self.csv_logger)
    def predict(self, x):
        self.model.predict(x)
def save_vn(model, fname="vn.test.model"):
    model.updater=[]
    model.callbacks = []
    model.lstm_model = []
    model.selector = []
    pickle.dump(model, open(fname,'wb'))
def load_vn(model_fname, selector_h5, predictor_h5):
    model = pickle.load(open(model_fname, "rb"))
    model.predictor = []
    model.selector = []
    model.LSTM = []
    model.callbacks=[]
    return model
if __name__ == "__main__":
    # a = voltnet_model(pred_str=0)
    # a.fit_from_prob()
    # a.save()
    pred_str_list = [5,10,20,40]
    io_dir = "F:\\lstm_data\\"
    file_base_name = "Scaled_lstm_2c.h5.str"
    train_size = 10000
    test_size = 3000
    for pred_str in pred_str_list:
        load_file = io_dir + file_base_name + str(pred_str)
        with h5py.File(load_file,'r') as f:
            tag = f["y"][:train_size + test_size]
            x = f["x"][:train_size + test_size,-34:,...]
        lstm = voltnet_LSTM(pred_str=pred_str)
        lstm.fit(x=x[:train_size,...], y=tag[:train_size], x_test=x[train_size:train_size+test_size,...], y_test=tag[train_size:train_size+test_size,...])