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
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
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
from os import path
def load_h5(fname, categorical=True, cut_trace=True, trace_len=50, load_size=20000):
    if cut_trace:
        start = trace_len
    else:
        start = 0
    with h5py.File(fname,'r') as f:
        tag = f["y"][:load_size]
        x = f["x"][:load_size,-start:,...]
    if categorical:
        tag = to_categorical(tag[:load_size])
    return [x, tag]
class voltnet_model():
    def __init__(self, name="voltnet", pred_str=0, trace_len=50):
        self.name = name
        self.pred_str = pred_str # this is only used for naming. The actual configuration for pred_str is fixed during data generation
        self.LSTM = voltnet_LSTM(pred_str=self.pred_str, trace_len=trace_len)
        self.callbacks = []
        self.prepare_callback_prune()
        self.prepare_callback_early()
        self.len = trace_len
        self.train_size = 10000
        self.test_size = 3000
        self.data_size = 120000
        self.validation_split = 0.3
        self.lstm_model = "voltnet.selector.str." + str(self.pred_str) +"." + str(self.len) + ".hdf5"
    def load_LSTM(self):
        self.LSTM = load_frozen_lstm(self.lstm_model)
    def load_h5(self, fname, categorical=True, cut_trace=True):
        # This method should NOT be called anymore
        if cut_trace:
            start = self.len
        else:
            start = 0
        with h5py.File(fname,'r') as f:
            tag = f["y"][:self.data_size]
            x = f["x"][:self.data_size,-start:,...]
        if categorical:
            tag = to_categorical(tag[:self.data_size])
        return [x, tag]
    def fit(self, lstm_train_data="Scaled_lstm_2c.h5.0.25", grid_train_data="Scaled_VoltNet_2c.h5", prob_fname='F:\\lstm_data\\prob_distribution.h5'):
        [x, y] = load_h5(fname=lstm_train_data, categorical=False, trace_len=self.len, load_size=self.data_size)
        self.LSTM.fit(x, y)
        sweeper = lstm_sweep(lstm_model=self.LSTM.filepath, scaled_grid_fname=grid_train_data, save_fname=prob_fname, trace_len=self.len)
        sweeper.process()
        print("Sweeping completed")
        self.fit_from_prob(fname=prob_fname)
    def fit_from_lstm(self, lstm_model, grid_train_data="Scaled_VoltNet_2c.h5", prob_fname='F:\\lstm_data\\prob_distribution.h5'):
        sweeper = lstm_sweep(lstm_model=lstm_model, scaled_grid_fname=grid_train_data, save_fname=prob_fname)
        #sample_size = (self.train_size + self.test_size) * 1.5
        sweeper.process()
        self.fit_from_prob(fname=prob_fname)
    def fit_from_prob(self, fname='F:\\lstm_data\\prob_distribution.h5'):
        [x, y] = load_h5(fname=fname, cut_trace=False, categorical=False, trace_len=self.len, load_size=self.data_size)
        self.fit_selector_(x=x, y=y)
        self.fit_predictor_(x=x, y=y)
    def fit_selector_(self, x, y):
        inputs = Input(shape=(x.shape[1],))
        outputs = sparsity.prune_low_magnitude(Dense(1, activation='sigmoid'), **self.pruning_params)(inputs)
        self.selector = keras.models.Model(inputs=inputs, outputs=outputs)
        self.selector.summary()
        self.selector.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.selector.fit(x, y,
            validation_split=self.validation_split,
            batch_size=32,
            epochs=15,
            shuffle='batch',
            callbacks=self.callbacks,
            verbose=1)
    def fit_predictor_(self, x, y):
        output_weights = self.selector.layers[-1].get_weights()[0]
        weight_norm = np.linalg.norm(output_weights, axis=1)
        self.selected_sensors = weight_norm > 0.3
        print(np.sum(self.selected_sensors))
        selected_x = x[:,self.selected_sensors]
        n = 64
        inputs = Input(shape=(selected_x.shape[1]))
        selu_ini = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/34, seed=None)
        outputs = Dense(n, activation='selu', kernel_initializer=selu_ini, bias_initializer='zeros')(inputs)
        outputs = BatchNormalization()(outputs)
        outputs = Dense(1, activation='sigmoid')(inputs)
        self.predictor = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        self.predictor.summary()
        ck_point = "predictor"+ str(self.name) + ".hdf5"#+ ".{epoch:02d}-{val_categorical_accuracy:.3f}-{val_loss:.3f}.hdf5"
        self.predictor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.predictor.fit(selected_x, y,
            shuffle='batch',
            validation_split=self.validation_split,
            batch_size=1,
            epochs=15,
            callbacks=[ CSVLogger('voltnet.csv',append=True),
                        ModelCheckpoint(ck_point, monitor='val_loss',save_best_only=True, verbose=1, mode='min'),
                        #self.early,
                        ],
            verbose=1)
    def save(self, selection_fname):
        #self.predictor.save('voltnet.predictor.h5')
        pickle.dump(self.selected_sensors, open(selection_fname,'wb'))
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
        self.ckp = ModelCheckpoint(filepath, monitor='val_loss',save_best_only=True, verbose=1, mode='min')
        self.callbacks.append(self.ckp)
    def prepare_callback_csv(self):
        self.csv_logger = CSVLogger('pruned_training.csv',append=True)
        self.callbacks.append(self.csv_logger)
    def prepare_callback_early(self):
        self.early = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
        #self.callbacks.append(self.early)
    def predict(self, x):
        result = self.predictor.predict(x[:,self.selected_sensors])
        #print(result)
        return result >= 0.5
        #return int(np.argmax(result, axis=1))
    def evaluate(self, x, y):
        X = x[:,self.selected_sensors]
        y_pred = self.predictor.predict(X)
        return [0, mean_squared_error(y, y_pred)]
class voltnet_LSTM():
    def __init__(self, pred_str=0, trace_len=50):
        self.pred_str = pred_str
        self.callbacks = []
        self.len = trace_len
        self.prepare_callback_ckp()
        self.prepare_callback_csv()
        self.prepare_callback_early()
        self.validation_split = 0.3
    def fit(self, x, y):
        if x.shape[1] != self.len:
            x = x[:,-self.len:,...]
        # NN hyperparameter
        rnn_dropout = 0.5
        m = 32
        s = 3
        n = 50
        batch_size = 64
        # setting up lstm
        inputs = Input(shape=(self.len,1))
        for rnn in range(s):
            if rnn == 0:
                lstm = Bidirectional(LSTM(n, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=True))(inputs)
                node = Add()([inputs, lstm])
                #node = lstm
            elif rnn < s - 1:
                lstm = Bidirectional(LSTM(n, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=True))(node)
                node = Add()([node, lstm])
                #node = lstm
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
        [xv, yv] = load_h5(fname="F:\\facesim2c.lstm.str5.scaled", cut_trace=True, categorical=False, trace_len=self.len, load_size=4000)
        self.model.fit(x, y,
            #validation_split=self.validation_split,
            validation_data=(xv,yv),
            batch_size=batch_size,
            shuffle="batch",
            epochs=13,
            callbacks=self.callbacks,
            verbose=1)
    def prepare_callback_clr(self):
        self.clr = CyclicLR(base_lr=0.05, max_lr=0.15, mode='triangular2')
        self.callbacks.append(self.clr)
    def prepare_callback_ckp(self):
        self.filepath = "voltnet.lstm.str." + str(self.pred_str) +".len" + str(self.len) + ".hdf5"#+ ".{epoch:02d}-{val_acc:.3f}-{val_loss:.3f}.hdf5"
        self.checkpoint = ModelCheckpoint(self.filepath, monitor='val_loss',save_best_only=True, verbose=1, mode='min')
        self.callbacks.append(self.checkpoint)
    def prepare_callback_early(self):
        self.early = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
       # self.callbacks.append(self.early)
    def prepare_callback_csv(self, fname='training_residual.csv'):
        try:
            os.remove(fname)
        except:
            pass
        self.csv_logger = CSVLogger(fname, append=True)
        self.callbacks.append(self.csv_logger)
    # def predict(self, x, batch_size=None):
    #     pred = self.model.predict(x, batch_size=None)
    #     return pred
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
    lstm_load_list = [
            path.join("F:\\","blackscholes2c" + ".lstm"),
            path.join("F:\\","bodytrack2c" + ".lstm"),
            path.join("F:\\","freqmine2c" + ".lstm"),
            path.join("F:\\","facesim2c" + ".lstm"),
    ]
    net_load_list = [
            path.join("F:\\","blackscholes2c" + ".voltnet"),
            path.join("F:\\","bodytrack2c" + ".voltnet"),
            path.join("F:\\","freqmine2c" + ".voltnet"),
            path.join("F:\\","facesim2c" + ".voltnet"),
    ]
    pred_str_list = [20,10,5,0]
    trace_len_list = [ 40, 30, 20,50]

    for pred_str in pred_str_list:
        for trace_len in trace_len_list:
                lstm_load = lstm_load_list[0]
                net_load = net_load_list[0]
                str_tag = ".str" + str(pred_str)
                len_tag = ".len" + str(trace_len)
                balancing_lstm_load =  lstm_load + str_tag
                balancing_grid_load =  net_load + str_tag
                scaled_grid_fname = balancing_grid_load + ".scaled"
                scaled_lstm_fname = balancing_lstm_load + ".scaled"
                prob_fname = balancing_grid_load + len_tag + ".prob"

                model_tag = str_tag + len_tag
                voltnet = voltnet_model(name=model_tag, pred_str=pred_str, trace_len=trace_len)
                #voltnet.fit(lstm_train_data=scaled_lstm_fname, grid_train_data=scaled_grid_fname, prob_fname=prob_fname)
                #voltnet.fit_from_lstm(lstm_model="voltnet.lstm.str.5.len50.hdf5", grid_train_data=scaled_grid_fname, prob_fname=prob_fname)
                voltnet.fit_from_prob(fname=prob_fname)

                save_file = "voltnet.final" + model_tag + ".pk"
                voltnet.save(save_file)
    # io_dir = "F:\\lstm_data\\"
    # grid_base_name = "Scaled_VoltNet_2c.str"
    # trac_base_name = "Scaled_lstm_2c.h5.str"
    # prob_base_name = "prob_distribution.str"
    # lstm_base_name = "voltnet.selector.str."
    # save_base_name = "voltnet.selected.str."
    # for pred_str in pred_str_list:
    #     grid_file = io_dir + grid_base_name + str(pred_str)
    #     trac_file = io_dir + trac_base_name + str(pred_str)
    #     prob_file = io_dir + prob_base_name + str(pred_str)
    #     lstm_file = lstm_base_name + str(pred_str) + ".hdf5"
    #     save_file = save_base_name + str(pred_str) + ".pk"

    #     voltnet = voltnet_model(name="str"+str(pred_str))
    #     voltnet.fit_from_lstm(lstm_model="voltnet.selector.str.0.19-0.994-0.023.hdf5",  grid_train_data=grid_file, prob_fname=prob_file)
    #     voltnet.save(selection_fname=save_file)
    import winsound
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    #file_base_name = "Scaled_lstm_2c.h5.str"
    # train_size = 200000
    # test_size = 3000
    # for pred_str in pred_str_list:
    #     load_file = io_dir + file_base_name + str(pred_str)
    #     with h5py.File(load_file,'r') as f:
    #         tag = f["y"][:train_size + test_size]
    #         x = f["x"][:train_size + test_size,...]
    #     lstm = voltnet_LSTM(pred_str=pred_str)
    #     lstm.fit(x=x[:train_size,...], y=tag[:train_size], x_test=x[train_size:train_size+test_size,...], y_test=tag[train_size:train_size+test_size,...])