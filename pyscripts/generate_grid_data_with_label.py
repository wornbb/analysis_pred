import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py
import numpy as np
from loading import generate_prediction_data
from tensorflow.keras.utils import to_categorical
import os
if os.name == 'nt':
      fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"
else:
      fname = "/data/yi/voltVio/analysis/raw/blackscholes2c.gridIR"
(data, tag) = generate_prediction_data(fname,lines_to_read=1000000, selected_sensor='all',trace=45)
print("Loading complete")
data = np.moveaxis(data, 2, 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
for instance in range(data.shape[1]):
      piece = data[:,instance,:]
      data[:,instance,:] = scaler.fit_transform(piece)
data = np.expand_dims(data, axis=-1)

with h5py.File("balanced_grid_sensor.h5","w") as hf:
      hf.create_dataset("data", data=data, dtype = 'float32')
      hf.create_dataset("tag", data=tag, dtype = 'float32')