import tensorflow as tf
import numpy as np
from loading import generate_prediction_data
from clr_callback import*
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import TimeDistributed, Input, Dense, BatchNormalization, Flatten
from tensorflow.keras.layers import Dense, Dropout,Add, LSTM, Bidirectional, BatchNormalization, Input, Permute
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from keras.utils.io_utils import HDF5Matrix

from sklearn.metrics import make_scorer, mean_squared_error
import pickle
from loading import *
from clr_callback import*
import h5py
import os
class voltnet_model():
    def __init__(self):
        self.LSTM = voltnet_LSTM()
        self.callbacks = []
        self.prepare_callback_prune()
    def load_LSTM(self, model_name='residual.3.biLSTM.45.15-0.997-0.008.hdf5'):
        self.LSTM = load_frozen_lstm(model_name)
    def load_prob(self, fname):
        train_size = 20000
        test_size = 1000
        with h5py.File(fname,'r') as f:
            tag = f["y"][()]
            x = f["x"][:train_size+test_size,...]
        tag = to_categorical(tag[:train_size+test_size])
        
        #self.x_train = HDF5Matrix(datapath=fname, dataset='x', start=0, end=train_size)
        #self.x_test = HDF5Matrix(datapath=fname, dataset='x', start=train_size, end=train_size+test_size)
        self.x_train = x[:train_size,...]
        self.x_test = x[train_size:train_size+test_size,...]
        self.y_train = tag[:train_size,:]
        self.y_test = tag[train_size:train_size+test_size,:]
    def fit_from_prob(self, fname='F:\\lstm_data\\prob_distribution.h5'):
        self.load_prob(fname=fname)
        self.fit_selector_()
        self.fit_predictor_()
    def fit_selector_(self):
        inputs = Input(shape=(self.x_train.shape[1],))
        outputs = sparsity.prune_low_magnitude(Dense(2, activation='softmax'), **self.pruning_params)(inputs)
        self.selector = keras.models.Model(inputs=inputs, outputs=outputs)
        self.selector.summary()
        self.selector.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        self.selector.fit(self.x_train, self.y_train,
            validation_data=(self.x_test,self.y_test),
            batch_size=32,
            epochs=30,      
            shuffle='batch',
            callbacks=self.callbacks,
            verbose=1)
    def fit_predictor_(self):
        output_weights = self.selector.layers[-1].get_weights()[0]
        weight_norm = np.linalg.norm(output_weights, axis=1)
        self.selected_sensors = weight_norm > 0.1
        x_train = self.x_train[:,self.selected_sensors]
        x_test = self.x_test[:, self.selected_sensors]

        n = 300
        inputs = Input(shape=(x_train.shape[1]))
        selu_ini = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/34, seed=None)
        outputs = Dense(n, activation='selu', kernel_initializer=selu_ini, bias_initializer='zeros')(inputs)
        outputs = BatchNormalization()(outputs)
        outputs = Dense(2, activation='softmax')(inputs)
        self.predictor = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        self.predictor.summary()
        ck_point = "pruned.nn."+ str(n) + ".{epoch:02d}-{val_categorical_accuracy:.3f}-{val_loss:.3f}.hdf5"
        self.predictor.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        self.predictor.fit(x_train, self.y_train,
            shuffle='batch',
            validation_data=(x_test[:,:],self.y_test[:,:]),
            batch_size=1,
            epochs=1,      
            callbacks=[ CSVLogger('pruned_training.csv',append=True),
                        ModelCheckpoint(ck_point, monitor='val_categorical_accuracy',save_best_only=True, verbose=1, mode='max'),
                        ],
            verbose=1)
        # def fit(self, data_train):
        #     output_weights = self.selector.layers[-1].get_weights()[0]
        #     weight_norm = np.linalg.norm(output_weights, axis=1)
        #     selected_sensors = weight_norm > 0.1
        #     x_train = x_train[:,selected_sensors]
        #     x_test = x_test[:, selected_sensors]
        #     for n in range(300,301,5):
        #           inputs = Input(shape=(x_train.shape[1]))
        #           selu_ini = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/34, seed=None)
        #           # outputs = Dense(n, activation='selu', kernel_initializer=selu_ini, bias_initializer='zeros')(inputs)
        #           # outputs = BatchNormalization()(outputs)
        #           # outputs = Dense(n/2, activation='selu', kernel_initializer=selu_ini, bias_initializer='zeros')(inputs)
        #           # outputs = BatchNormalization()(outputs)
        #           outputs = Dense(2, activation='softmax')(inputs)
        #           model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        #           model.summary()
        #           ad = tf.keras.optimizers.Adam(lr=0.03, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #           logdir = 'prune.log'
        #           ck_point = "pruned.nn."+ str(n) + ".{epoch:02d}-{val_categorical_accuracy:.3f}-{val_loss:.3f}.hdf5"
        #           model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        #           model.fit(x_train[:,:], y_train[:,:],
        #                 shuffle='batch',
        #                 validation_data=(x_test[:,:],y_test[:,:]),
        #                 batch_size=1,
        #                 epochs=200,      
        #                 callbacks=[ CSVLogger('pruned_training.csv',append=True),
        #                             ModelCheckpoint(ck_point, monitor='val_categorical_accuracy',save_best_only=True, verbose=1, mode='max'),
        #                             ],
        #                 verbose=1)

    def prepare_callback_prune(self):
        self.pruning_params = {
            'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                        final_sparsity=0.90,
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
        self.predictor.predict(x[:,self.selected_sensors])
    def evaluate(self, x, y):
        X = x[:,self.selected_sensors]
        y_pred = self.predictor.predict(X)
        return [0, mean_squared_error(y, y_pred)]
class voltnet_LSTM():
    def __init__(self):
        self.callbacks = []
        self.prepare_callback_ckp()
        self.prepare_callback_csv()
    def fit(self, x, y):
        # NN hyperparameter
        rnn_dropout = 0.4
        m = 32
        s = 4
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
        selu_ini = keras.initializers.RandomNormal(mean=0.0, stddev=1/40, seed=None)
        node = Dense(m, activation='selu', kernel_initializer=selu_ini)(node)
        node = BatchNormalization()(node)
        outputs = Dense(1, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros')(node)
        self.model = keras.models.Model(inputs=inputs, outputs=outputs)
        rmsprop = keras.optimizers.rmsprop(lr=0.005)
        self.model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
        self.model.summary()
        print('Train...')  
        self.model.fit(x, y,
            batch_size=batch_size,
            shuffle="batch",
            epochs=15,
            callbacks=self.callbacks,
            verbose=1)
    def prepare_callback_clr(self):
        self.clr = CyclicLR(base_lr=0.05, max_lr=0.15, mode='triangular2')
        self.callbacks.append(self.clr)
    def prepare_callback_ckp(self):
        filepath = "voltnet." + ".selector."  + ".{epoch:02d}-{val_acc:.3f}-{val_loss:.3f}.hdf5"
        self.checkpoint = ModelCheckpoint(filepath, monitor='val_acc',save_best_only=True, verbose=1, mode='max')
        self.callbacks.append(self.checkpoint)
    def prepare_callback_csv(self, fname='training_residual.csv'):
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
    model.predictor = []
    model.selector = []
    model.callbacks = []
    model.LSTM = []
    pickle.dump(model, open(fname,'wb'))
def load_vn(model_fname, selector_h5, predictor_h5):
    model = pickle.load(open(model_fname, "rb"))
    model.predictor = []
    model.selector = []
    model.LSTM = []
    model.callbacks=[]
    return model
if __name__ == "__main__":
    a = voltnet_model()
    a.fit_from_prob()
    save_vn(a)