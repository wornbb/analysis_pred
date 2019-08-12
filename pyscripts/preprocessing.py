import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py
import numpy as np
import pickle
from loading import generate_prediction_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from clr_callback import*
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.optimizers import *
from sklearn.ensemble import BaggingRegressor

import os

from sklearn.preprocessing import MinMaxScaler


class preprocessor():
    """scale the traced_grid and LSTM trace data for training
    Method:
        scale_grid_trace: process traced grid
    Attribute:
    """
    def __init__(self, load_fname, save_fname):
        self.load_fname = load_fname
        self.save_fname = save_fname
        self.scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        with h5py.File(self.load_fname, 'r') as f:
            self.load_x = f["x"][()]
            self.load_y = f["x"][()]

    def scale_grid_trace(self):
        self.loaded_samples = self.load_x.shape[0]
        self.trace_len = self.load_x.shape[2]
        self.grid_size = self.load_x.shape[1]
        with h5py.File(self.save_fname, 'w') as f:
            self.save_x = f.create_dataset("x", shape=self.load_x.shape)
            self.save_y = f.create_dataset("y",data=self.load_y)
            for sample in range(self.loaded_samples):
                self.save_x[sample,:,:,0] = self.scaler.fit_transform(self.load_x[sample,:,:,0])
    def sacle_lstm(self):
        x = np.squeeze(x, axis=(2))
        y = y.flatten().astype('int')
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
    import os
    if os.name == 'nt':
        load_fname = "VoltNet_2c.h5"
        save_fname = "Scaled_VoltNet_2c.h5"
    else:
        load_fname = "/media/yi/yi_final_resort/VoltNet_2c.h5"
        save_fname = "/media/yi/yi_final_resort/Scaled_VoltNet_2c.h5"
    #grid_processor = preprocessor(load_fname,save_fname)
    #grid_processor.scale_grid_trace()
    sensor_model = load_model(r'residual.4.biLSTM.45.10-0.951-0.140.hdf5')
    with h5py.File(save_fname,'r') as f:
        x = f["x"][()]
    for node in range(100):
        pred = sensor_model.predict(x[0,node:node+1,:,0:1])
        print(pred)
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
