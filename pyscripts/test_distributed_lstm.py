import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py
import numpy as np
from loading import generate_prediction_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from clr_callback import*
from tensorflow.keras.optimizers import *
with h5py.File('./balanced_grid_sensor.h5','r') as f:
      data = f["data"].value
      tag = f["tag"].value

sensor_model = load_model('lstm.25.nn.25.16-0.888-0.297.hdf5')

x = np.ones((data.shape[0],data.shape[1]))
for sample in range(data.shape[0]):
    x[sample,:] = sensor_model.predict(data[sample,:,6:-5,0:1], batch_size=data.shape[1])[:,0]

with h5py.File('./batch_lstm_result.h5', 'w') as hf:
      hf.create_dataset("x", data=x, dtype = 'float32')
      hf.create_dataset("tag", data=tag, dtype = 'float32')
# with h5py.File('./batch_lstm_result.h5', 'r') as f:
#       x = f["x"].value
#       tag = f["tag"].value
tag = to_categorical(tag)
[x_train, x_test] = np.array_split(x, 2, axis=0)
[y_train, y_test] = np.array_split(tag, 2, axis=0)


from tensorflow.keras.layers import TimeDistributed, Input, Dense, BatchNormalization, Flatten
inputs = Input(shape=(x.shape[1]))
node = Dense(1000, activation='selu')(inputs)
node = BatchNormalization()(node)
node = Dense(76, activation='selu')(inputs)
node = BatchNormalization()(node)
node = Dense(76, activation='selu')(inputs)
node = BatchNormalization()(node)
outputs = Dense(2, activation='softmax')(node)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
ad = tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(x_train[::31,:], y_train[::31,:],
      validation_data=(x_test[::91,:],y_test[::91,:]),
      batch_size=1,
      epochs=200,
      verbose=1)
