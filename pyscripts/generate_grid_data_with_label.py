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
else:
      #name = input("Give the simulation name: ")
      fname = "/data/yi/voltVio/analysis/raw/" + name + ".gridIR"
      #print("loading " + fname)
      f_list = [
      "/data/yi/voltVio/analysis/raw/" + "blackscholes2c" + ".gridIR",
      "/data/yi/voltVio/analysis/raw/" + "bodytrack2c" + ".gridIR",
      "/data/yi/voltVio/analysis/raw/" + "freqmine2c"+ ".gridIR",
      "/data/yi/voltVio/analysis/raw/" + "facesim2c"+ ".gridIR",
      ]
grid_size = 5776
lstm_samples = 0
grid_samples = 0

with h5py.File("lstm_2c.h5", 'w') as lstmF:
      lstmX = lstmF.create_dataset('x', shape=(1, 34, 1), maxshape=(None, 34, 1))
      lstmY = lstmF.create_dataset('y', shape=(1,), maxshape=(None,))
      with h5py.File("VoltNet_2c.h5", 'w') as netF:
            netX = netF.create_dataset('x', shape=(1, grid_size, 34, 1), maxshape=(None, grid_size, 34, 1))
            netY = netF.create_dataset('y', shape=(1,), maxshape=(None,))
            for fname in f_list:
                  [lstm_data, lstm_tag, grid_data, gird_tag] = generate_prediction_data(fname, selected_sensor='all',trace=39, ref=1, pred_str=5, thres=4)
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



print("Loading complete")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
for instance in range(data.shape[1]):
      piece = data[:,instance,:]
      data[:,instance,:] = scaler.fit_transform(piece)
data = np.expand_dims(data, axis=-1)

save_fname = "balanced_gird_sensor." + name + ".h5"
print("Saving to " + save_fname)
with h5py.File(save_fname,"w") as hf:
      hf.create_dataset("x", data=data, dtype = 'float32')
      hf.create_dataset("y", data=tag, dtype = 'float32')