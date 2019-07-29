import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py
import numpy as np
from loading import generate_prediction_data
from tensorflow.keras.utils import to_categorical
import os
if os.name == 'nt':
      fname = "F:\\Yaswan2c\\Yaswan2c.gridIR"
else:
      fname = "/data/yi/voltVio/analysis/raw/freqmine2c.gridIR"
(data, tag) = generate_prediction_data(fname, selected_sensor='all',trace=39, ref=1, pred_str=5, thres=4, lines_to_read=10000)
print("Loading complete")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
for instance in range(data.shape[1]):
      piece = data[:,instance,:]
      data[:,instance,:] = scaler.fit_transform(piece)
data = np.expand_dims(data, axis=-1)

with h5py.File("old_balanced_grid_sensor.h5","w") as hf:
      hf.create_dataset("data", data=data, dtype = 'float32')
      hf.create_dataset("tag", data=tag, dtype = 'float32')