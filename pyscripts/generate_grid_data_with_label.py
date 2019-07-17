import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from loading import generate_prediction_data
from tensorflow.keras.utils import to_categorical

fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"
(data, tag) = generate_prediction_data(fname, selected_sensor='all',trace=40)
data = np.moveaxis(data, 2, 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
for instance in range(data.shape[0]):
      data[instance,:,:] = scaler.fit_transform(data[instance,:,:])
      data = np.expand_dims(data, axis=-1)

pickle.dump([data, tag], open('balanced_grid_sensor.data', 'wb'))