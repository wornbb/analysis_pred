import os
import pickle

import h5py
import numpy as np
import tensorflow as tf
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical
from tensorflow_model_optimization.sparsity import keras as sparsity

from clr_callback import *
from loading import *
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from os import path
class MyScaler():
    def fit_transform(self,x):
        transformed = (x - 1) * 100
        return transformed
class lstm_sweep():
    """ process traced grid to produce grid of probabilities.
        Only process 1 file at a time, because we assume the data loaded is already combined.
    """
    def __init__(self, lstm_model='residual.4.biLSTM.45.10-0.951-0.140.hdf5', scaled_grid_fname=[], save_fname=[], trace_len=50, mode="soft"):
        self.unlock = True
        self.len = trace_len
        self.load_fname = scaled_grid_fname
        self.save_fname = save_fname
        self.mode = mode
        self.open_for_write()
        if self.unlock is True:
            self.lstm_model = load_frozen_lstm(lstm_model)
            self.open_for_read()
    def open_for_write(self):
        if self.mode == "soft" and path.exists(self.save_fname):
            self.unlock = False
        else:
            self.sf = h5py.File(self.save_fname, 'w')
            self.saveX = self.sf.create_dataset("x", shape=(self.data_shape[0], self.data_shape[1]), chunks=(1,self.data_shape[1]))
            self.saveY = self.sf.create_dataset("y", data=self.y)
    def open_for_read(self):
        self.lf = h5py.File(self.load_fname, 'r')
        self.x = self.lf["x"]
        self.y = self.lf["y"]
        self.data_shape = self.x.shape
        #self.samples = int(self.data_shape[0] * 0.1)
    def process(self, samples=None):
        if self.unlock:
            if samples == None:
                samples = self.data_shape[0]
            for sample in range(samples):
                self.saveX[sample,:] = self.lstm_model.predict(self.x[sample,:,-self.len:,...], batch_size=self.data_shape[1]).flatten()
                if sample % 10000 == 0:
                    self.sf.flush()
            self.sf.close()
            self.lf.close()
class preScaler():
    """scale the traced_grid and LSTM trace data for training
    Method:
        scale_grid_trace: process traced grid
    Attribute:
    """
    def __init__(self, load_fname, save_fname):
        self.load_fname = load_fname
        self.save_fname = save_fname
        self.scaler = MyScaler()
        #self.scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        with h5py.File(self.load_fname, 'r') as f:
            self.load_x = f["x"][()]
            self.load_y = f["y"][()]

    def scale_grid_trace(self):
        self.loaded_samples = self.load_x.shape[0]
        self.trace_len = self.load_x.shape[2]
        self.grid_size = self.load_x.shape[1]
        with h5py.File(self.save_fname, 'w') as f:
            self.save_x = f.create_dataset("x", shape=self.load_x.shape)
            self.save_y = f.create_dataset("y",data=self.load_y)
            for sample in range(self.loaded_samples):
                self.save_x[sample,:,:,0] = self.scaler.fit_transform(self.load_x[sample,:,:,0])
                f.flush()
    def scale_lstm(self):
        x = np.squeeze(self.load_x, axis=(2))
        y = self.load_y.flatten().astype('int')
        shuffle_index = np.arange(y.shape[0])
        np.random.shuffle(shuffle_index)
        x = x[shuffle_index,:]
        y = y[shuffle_index]
        scaled_x = self.scaler.fit_transform(x[:,:])
        scaled_x = np.expand_dims(scaled_x, axis=2)
        with h5py.File(self.save_fname, 'w') as f:
            self.save_x = f.create_dataset("x", data=scaled_x)
            self.save_y = f.create_dataset("y", data=y)
if __name__ == "__main__":
    if os.name == 'nt':
        load_dir = "F:\\"
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
        #pred_str_list = [0,5,10,20]#,40]
        pred_str_list = [15,25,30,35]
        lstm_model = 'voltnet..selector..16-0.961-0.130.hdf5'
        scaled_load_grid_file = "Scaled_VoltNet_2c.h5"
        prob_distribution_file = "F:\\lstm_data\\prob_distribution.h5"
    else:
        load_dir = "/media/yi/yi_final_resort/"
        lstm_load_list = [
                path.join(load_dir,"blackscholes2c" + ".lstm"),
                path.join(load_dir,"bodytrack2c" + ".lstm"),
                path.join(load_dir,"freqmine2c" + ".lstm"),
                path.join(load_dir,"facesim2c" + ".lstm"),
        ]
        net_load_list = [
                path.join(load_dir,"blackscholes2c" + ".voltnet"),
                path.join(load_dir,"bodytrack2c" + ".voltnet"),
                path.join(load_dir,"freqmine2c" + ".voltnet"),
                path.join(load_dir,"facesim2c" + ".voltnet"),
        ]
        #pred_str_list = [0,5,10,20]#,40]
        pred_str_list = [15,25,30,35]
        lstm_model = 'voltnet..selector..16-0.961-0.130.hdf5'
        scaled_load_grid_file = "Scaled_VoltNet_2c.h5"
        prob_distribution_file = "F:\\lstm_data\\prob_distribution.h5"
    from PyInquirer import style_from_dict, Token, prompt
    def grid_task(load_f, save_f):
        grid_processor = preScaler(load_f, save_f)
        grid_processor.scale_grid_trace()
    def lstm_task(load_f, save_f):
        grid_processor = preScaler(load_f, save_f)
        grid_processor.sacle_lstm()
    questions = [
        {
            'type': 'checkbox',
            'message': 'Select the process for the data',
            'name': 'selection',
            'choices': [ 
                {
                    'name': 'Scale training grid trace',
                    'value': 1
                },
                {
                    'name': 'Scale training LSTM trace',
                    'value':2
                },
                {
                    'name': 'generate prob distribution',
                    'value': 3
                },
                {
                    'name': 'raw benchmarking lstm',
                    'value': 4
                },
                {
                    'name': 'All grid',
                    'value': 5
                },
                {
                    'name': 'All LSTM',
                    'value': 6
                },
            ],
            'validate': lambda answer: 'You must choose at least one process.' \
                if len(answer) == 0 else True
        }
    ]
    answers = prompt(questions)
    if 1 in answers['selection']:
        # only processing blackschole
        for pred_str in pred_str_list:
            balancing_lstm_load =  lstm_load_list[0] + ".str" + str(pred_str)
            balancing_grid_load =  net_load_list[0] + ".str" + str(pred_str)
            scaled_grid_save_fname = balancing_grid_load + ".scaled"
            scaled_lstm_save_fname = balancing_lstm_load + ".scaled"
            grid_processor = preScaler(balancing_grid_load,scaled_grid_save_fname)
            grid_processor.scale_grid_trace()
    if 2 in answers['selection']:
        # only processing blackschole
        for pred_str in pred_str_list:
            balancing_lstm_load =  lstm_load_list[0] + ".str" + str(pred_str)
            balancing_grid_load =  net_load_list[0] + ".str" + str(pred_str)
            scaled_grid_save_fname = balancing_grid_load + ".scaled"
            scaled_lstm_save_fname = balancing_lstm_load + ".scaled"
            grid_processor = preScaler(balancing_lstm_load, scaled_lstm_save_fname)
            grid_processor.scale_lstm()
    if 3 in answers['selection']:
        lstm_model = "voltnet.selector.str.0.len50.hdf5"
        scaled_load_grid_file = "F:\\blackscholes2c.voltnet.str0.scaled"
        prob_distribution_file = "F:\\blackscholes2c.voltnet.str0.len50.prob"
        prob_generator = lstm_sweep(lstm_model, scaled_load_grid_file, prob_distribution_file)
        prob_generator.process()
    if 4 in answers['selection']:
        sensor_model = load_model(lstm_model)
        # LSTM dataset: positive test
        with h5py.File(scaled_lstm_load_fname,'r') as f:
            x = f["x"][:10000,:,:]
            y = f["y"][:10000]
        pos_index = (y == 1)
        pos_x = x[pos_index, :, :]
        scores = sensor_model.evaluate(pos_x, y[pos_index], verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        # LSTM dataset: negative test
        # Grid dataset: overall test
        with h5py.File(scaled_grid_save_fname,'r') as f:
            x = f["x"][0,:100,:,:] 
            for node in range(100): 
                pred = sensor_model.predict(x[node:node+1,:,0:1])
                print(pred)
    if 5 in answers['selection']:
        # with Pool(processes=3) as pool: 
        #     results = pool.starmap(grid_task, zip(grid_load_fname_list, scaled_grid_save_fname_list), chunksize=3)
        for lstm_load, net_load in zip(lstm_load_list, net_load_list):
            for pred_str in pred_str_list:
                balancing_lstm_load =  lstm_load + ".str" + str(pred_str)
                balancing_grid_load =  net_load + ".str" + str(pred_str)
                scaled_grid_save_fname = balancing_grid_load + ".scaled"
                scaled_lstm_save_fname = balancing_lstm_load + ".scaled"
                grid_processor = preScaler(balancing_grid_load, scaled_grid_save_fname)
                grid_processor.scale_grid_trace()
    if 6 in answers['selection']:
        # with Pool(processes=3) as pool: 
        #     results = pool.starmap(lstm_task, zip(lstm_load_fname_list, scaled_lstm_load_fname_list), chunksize=3)
        for lstm_load, net_load in zip(lstm_load_list, net_load_list):
            for pred_str in pred_str_list:
                balancing_lstm_load =  lstm_load + ".str" + str(pred_str)
                balancing_grid_load =  net_load + ".str" + str(pred_str)
                scaled_grid_save_fname = balancing_grid_load + ".scaled"
                scaled_lstm_save_fname = balancing_lstm_load + ".scaled"
                grid_processor = preScaler(balancing_lstm_load, scaled_lstm_save_fname)
                grid_processor.scale_lstm()
# f_list = [r"balanced_gird_sensor.Yaswan2c_desktop.h5"]
# with h5py.File(f_list[0], 'r') as f:
#       x_shape = f["x"].shape
#       x_type = f["x"].dtype
#       y_type = f["y"].dtype
# print(x_shape)
# sensor_model = load_model(r'residual.3.biLSTM.45.15-0.997-0.008.hdf5')
# with h5py.File(r"combined_2c_gird_probability2.h5", 'w') as f:
#       maxshape = (None, x_shape[1])
#       probs = f.create_dataset('x', shape=(1, x_shape[1]), maxshape=maxshape, dtype=x_type)
#       y = f.create_dataset('y', shape=(1,), maxshape=(None,), dtype=y_type)
#       sample_count = 0
#       for fname in f_list:
#             # load unprocessed data
#             with h5py.File(fname, 'r') as dataset:
#                   grid_trace_x = dataset["x"][:]
#                   classes = dataset["y"][:]
#             # resize saving space
#             new_sample_count = sample_count + classes.shape[0]
#             probs.resize(new_sample_count, axis=0)
#             y.resize(new_sample_count, axis=0)
#             # writing
#             y[sample_count:new_sample_count] = classes
#             for sample in range(grid_trace_x.shape[0]):
#                   probs[sample_count + sample, :] = sensor_model.predict(grid_trace_x[sample,:,:,0:1], batch_size=grid_trace_x.shape[1])[:,0]
#             sample_count = new_sample_count
#             print(fname)
# with h5py.File('combined_2c_gird_probability2.h5', 'r') as f:
#       x = f["x"][()]
#       tag = f["y"][()]
# # with h5py.File(f_list[0], 'r') as dataset:
# #                   grid_trace_x = dataset["data"][:]
# #                   classes = dataset["tag"][:]
# tag = np.bitwise_not(tag < 1.5)
# tag = to_categorical(tag)
# [x_train, x_test] = np.array_split(x, 2, axis=0)
# [y_train, y_test] = np.array_split(tag, 2, axis=0)
