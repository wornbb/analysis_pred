import tensorflow as tf
from keras.models import load_model
import h5py
import numpy as np
import pickle
from loading import generate_prediction_data
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger
from clr_callback import*
from tensorflow_model_optimization.sparsity import keras as sparsity
from keras.optimizers import *
from keras.utils.io_utils import HDF5Matrix
from keras.layers import TimeDistributed, Input, Dense, BatchNormalization, Flatten

import keras
import os
# f_list = [
# "balanced_gird_sensor." + "blackscholes2c" + ".h5",
# "balanced_gird_sensor." + "bodytrack2c" + ".h5",
# "balanced_gird_sensor." + "freqmine2c"+ ".h5",
# "balanced_gird_sensor." + "facesim2c"+ ".h5",
# ]
# f_list = [r"balanced_gird_sensor.Yaswan2c_desktop.h5"]
# # with h5py.File(f_list[0], 'r') as f:
# #       x_shape = f["data"].shape
# #       x_type = f["data"].dtype
# #       y_type = f["tag"].dtype
# with h5py.File(f_list[0], 'r') as f:
#       x_shape = f["x"].shape
#       x_type = f["x"].dtype
#       y_type = f["y"].dtype
# print(x_shape)
# sensor_model = load_model(r'residual.4.biLSTM.45.10-0.951-0.140.hdf5')
# with h5py.File(r"combined_2c_gird_probability2.h5", 'w') as f:
#       maxshape = (None, x_shape[1])
#       probs = f.create_dataset('x', shape=(1, x_shape[1]), maxshape=maxshape, dtype=x_type)
#       y = f.create_dataset('y', shape=(1,), maxshape=(None,), dtype=y_type)
#       sample_count = 0
#       for fname in f_list:
#             # load unprocessed data
#             with h5py.File(fname, 'r') as dataset:
#                   grid_trace_x = dataset["x"][()]
#                   classes = dataset["y"][()]
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
prob_distribution = 'F:\\lstm_data\\prob_distribution.h5'
train_size = 10000
test_size = 1000
x_train = HDF5Matrix(datapath=prob_distribution, dataset='x', start=0, end=train_size)
x_test = HDF5Matrix(datapath=prob_distribution, dataset='x', start=train_size, end=train_size+test_size)
with h5py.File(prob_distribution, 'r') as f:
      tag = f["y"][()]
# with h5py.File(f_list[0], 'r') as dataset:
#                   grid_trace_x = dataset["data"][:]
#                   classes = dataset["tag"][:]
tag = to_categorical(tag[:train_size+test_size])
y_train = tag[:train_size,:]
y_test = tag[train_size:train_size+test_size,:]
#[x_train, x_test] = np.array_split(x, 2, axis=0)
#[y_train, y_test] = np.array_split(tag, 2, axis=0)

pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.99,
                                                   begin_step=100,
                                                   end_step=300,
                                                   frequency=100)
}
inputs = Input(shape=(x_train.shape[1],))
node = Dense(76, activation='sigmoid')(inputs)

# outputs = sparsity.prune_low_magnitude(Dense(76, activation='sigmoid'), **pruning_params)(inputs)
outputs = BatchNormalization()(node)
outputs = Dense(2, activation='softmax')(inputs)
# node = Dense(76, activation='selu')(inputs)
# node = BatchNormalization()(outputs)
# outputs = Dense(2, activation='softmax')(outputs)
model = keras.models.Model(inputs=inputs, outputs=outputs)
model.summary()
ad = tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
logdir = 'prune.log'

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(x_train, y_train,
      validation_data=(x_test,y_test),
      batch_size=32,
      epochs=30,      
      shuffle='batch',
      # callbacks=[  sparsity.UpdatePruningStep(),
      #             sparsity.PruningSummaries(log_dir=logdir, profile_batch=0),
      #             CSVLogger('pruned_training.csv',append=True),
      #             ],
      verbose=1)
                  #ModelCheckpoint(filepath, monitor='val_categorical_accuracy',save_best_only=True, verbose=1, mode='max'),
output_weights = model.layers[-1].get_weights()[0]
weight_norm = np.linalg.norm(output_weights, axis=1)
pickle.dump(weight_norm, open('weight_norm.pk','wb'))
# selected_sensors = weight_norm > 0.1
# x_train = x_train[:,selected_sensors]
# x_test = x_test[:, selected_sensors]
# for n in range(300,301,5):
#       inputs = Input(shape=(x_train.shape[1]))
#       selu_ini = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/34, seed=None)
#       # outputs = Dense(n, activation='selu', kernel_initializer=selu_ini, bias_initializer='zeros')(inputs)
#       # outputs = BatchNormalization()(outputs)
#       # outputs = Dense(n/2, activation='selu', kernel_initializer=selu_ini, bias_initializer='zeros')(inputs)
#       # outputs = BatchNormalization()(outputs)
#       outputs = Dense(2, activation='softmax')(inputs)
#       model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
#       model.summary()
#       ad = tf.keras.optimizers.Adam(lr=0.03, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#       logdir = 'prune.log'
#       ck_point = "pruned.nn."+ str(n) + ".{epoch:02d}-{val_categorical_accuracy:.3f}-{val_loss:.3f}.hdf5"
#       model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
#       model.fit(x_train[:,:], y_train[:,:],
#             shuffle='batch',
#             validation_data=(x_test[:,:],y_test[:,:]),
#             batch_size=1,
#             epochs=200,      
#             callbacks=[ CSVLogger('pruned_training.csv',append=True),
#                         ModelCheckpoint(ck_point, monitor='val_categorical_accuracy',save_best_only=True, verbose=1, mode='max'),
#                         ],
#             verbose=1)
