import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py
import numpy as np
from loading import generate_prediction_data
from tensorflow.keras.utils import to_categorical
import os
if os.name == 'nt':
      name = "Yaswan2c_desktop"
      fname = "F:\\Yaswan2c\\Yaswan2c.gridIR"
      f_list = [fname]
      lstm_save = "lstm_2c.h5"
      net_save = "VoltNet_2c.h5"
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


# with h5py.File(net_save, 'w') as netF:
#       netX = netF.create_dataset('x', shape=(1, grid_size, 34, 1), maxshape=(None, grid_size, 34, 1))
#       netY = netF.create_dataset('y', shape=(1,), maxshape=(None,))
for balance in balance_list:
      balance_list = [0.3, 0.25, 0.15, 0.1]
      sampled_lstm_save =  lstm_save + "." + str(balance)
      with h5py.File(sampled_lstm_save, 'w') as lstmF:
            lstmX = lstmF.create_dataset('x', shape=(1, 34, 1), maxshape=(None, 34, 1))
            lstmY = lstmF.create_dataset('y', shape=(1,), maxshape=(None,))
            for fname in f_list:
                  [lstm_data, lstm_tag, grid_data, gird_tag] = generate_prediction_data(fname, selected_sensor='all',trace=39, ref=1, pred_str=5, thres=4, balance=balance, grid_trigger=False)
                  new_lstm_samples = lstm_samples + lstm_data.shape[0]
                  new_grid_samples = grid_samples + grid_data.shape[0]
                  lstmX.resize(new_lstm_samples, axis=0)
                  lstmY.resize(new_lstm_samples, axis=0)
                  netX.resize(new_grid_samples, axis=0)
                  netY.resize(new_grid_samples, axis=0)
                  lstmX[lstm_samples:,:,0] = lstm_data
                  lstmY[lstm_samples:] = lstm_tag
                  netX[grid_samples:,:,:,0] = grid_data
                  netY[grid_samples:] = gird_tag
                  lstm_samples = new_lstm_samples
                  grid_samples = new_grid_samples
