import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py
import numpy as np
from loading import *
from tensorflow.keras.utils import to_categorical
import os
from loading import *
from os import path
if os.name == 'nt':
      name = "Yaswan2c_desktop"
      f_list = [
            path.join("F:\\gridIR","blackscholes2c" + ".gridIR"),
            path.join("F:\\gridIR","bodytrack2c" + ".gridIR"),
            path.join("F:\\gridIR","freqmine2c" + ".gridIR"),
            path.join("F:\\gridIR","facesim2c" + ".gridIR"),
      ]
      lstm_save = [
            path.join("F:\\","blackscholes2c" + ".lstm"),
            path.join("F:\\","bodytrack2c" + ".lstm"),
            path.join("F:\\","freqmine2c" + ".lstm"),
            path.join("F:\\","facesim2c" + ".lstm"),
      ]
      net_save = [
            path.join("F:\\","blackscholes2c" + ".voltnet"),
            path.join("F:\\","bodytrack2c" + ".voltnet"),
            path.join("F:\\","freqmine2c" + ".voltnet"),
            path.join("F:\\","facesim2c" + ".voltnet"),
      ]
else:
      f_list = [
      "/data/yi/voltVio/analysis/raw/" + "blackscholes2c" + ".gridIR",
      "/data/yi/voltVio/analysis/raw/" + "bodytrack2c" + ".gridIR",
      "/data/yi/voltVio/analysis/raw/" + "freqmine2c"+ ".gridIR",
      "/data/yi/voltVio/analysis/raw/" + "facesim2c"+ ".gridIR",
      ]
      dump = "/media/yi/yi_final_resort/"
      lstm_save = dump + "lstm_2c.h5"
      net_save = dump + "VoltNet_2c.h5"
grid_size = 5776
lstm_samples = 0
grid_samples = 0
balance_list = [0.1, 0.15, 0.25, 0.3]
# generating multiply training set with different pred_str
#pred_str_list = [0,5,10,20]#,40]
pred_str_list = [15,25,30,35]
#for fname,lstm_name,net_name in zip(f_list,lstm_save, net_save):
fname = f_list[0]
lstm_name = lstm_save[0]
net_name = net_save[0]
for pred_str in pred_str_list:
      balancing_lstm_save =  lstm_name + ".str" + str(pred_str)
      balancing_grid_save =  net_name + ".str" + str(pred_str)
      trace = 70 + pred_str
      generator = voltnet_training_data_factory([fname],trace=trace, ref=1, pred_str=pred_str, thres=4, pos_percent=0.25, grid_fsave=balancing_grid_save, lstm_fsave=balancing_lstm_save)
      generator.generate()
# # generating data for regression weakness plot
# balancing_grid_save =  dump + "overall_regression_testing.h5"
# generator = voltnet_training_data_factory(f_list,trace=10, ref=1, pred_str=0, thres=4, pos_percent=0.5, grid_fsave=balancing_grid_save, lstm_trigger=False)
# generator.generate()

# This is for generating normal data for training voltnet
# for balance in balance_list:
#       balancing_lstm_save =  lstm_save + "." + str(balance)
#       balancing_grid_save =  net_save + "." + str(balance)
#       generator = voltnet_training_data_factory(f_list,trace=39, ref=1, pred_str=5, thres=4, pos_percent=balance, grid_fsave=balancing_grid_save, lstm_fsave=balancing_lstm_save)
#       generator.generate()

# with h5py.File(net_save, 'w') as netF:
#       netX = netF.create_dataset('x', shape=(1, grid_size, 34, 1), maxshape=(None, grid_size, 34, 1))
#       netY = netF.create_dataset('y', shape=(1,), maxshape=(None,))
# balance_list = [0.3, 0.25, 0.15, 0.1]
# for balance in balance_list:
#       sampled_lstm_save =  lstm_save + "." + str(balance)
#       with h5py.File(sampled_lstm_save, 'w') as lstmF:
#             lstmX = lstmF.create_dataset('x', shape=(1, 34, 1), maxshape=(None, 34, 1))
#             lstmY = lstmF.create_dataset('y', shape=(1,), maxshape=(None,))
#             for fname in f_list:
#                   [lstm_data, lstm_tag, grid_data, gird_tag] = generate_prediction_data(fname, selected_sensor='all',trace=39, ref=1, pred_str=5, thres=4, balance=balance, grid_trigger=False)
#                   new_lstm_samples = lstm_samples + lstm_data.shape[0]
#                   new_grid_samples = grid_samples + grid_data.shape[0]
#                   lstmX.resize(new_lstm_samples, axis=0)
#                   lstmY.resize(new_lstm_samples, axis=0)
#                   netX.resize(new_grid_samples, axis=0)
#                   netY.resize(new_grid_samples, axis=0)
#                   lstmX[lstm_samples:,:,0] = lstm_data
#                   lstmY[lstm_samples:] = lstm_tag
#                   netX[grid_samples:,:,:,0] = grid_data
#                   netY[grid_samples:] = gird_tag
#                   lstm_samples = new_lstm_samples
#                   grid_samples = new_grid_samples
