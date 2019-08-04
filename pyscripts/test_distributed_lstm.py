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
with h5py.File('balanced_grid_sensor.h5','r') as f:
      data = f["data"].value
      tag = f["tag"].value

sensor_model = load_model('residual.3.biLSTM.45.15-0.997-0.008.hdf5')

x = np.ones((data.shape[0],data.shape[1]))
for sample in range(data.shape[0]):
    x[sample,:] = sensor_model.predict(data[sample,:,6:-5,0:1], batch_size=data.shape[1])[:,0]

with h5py.File('./batch_lstm_result.h5', 'w') as hf:
      hf.create_dataset("x", data=x, dtype = 'float32')
      hf.create_dataset("tag", data=tag, dtype = 'float32')
# with h5py.File('./batch_lstm_result.h5', 'r') as f:
#       x = f["x"].value
#       tag = f["tag"].value
tag = np.bitwise_not(tag < 1.5)
tag = to_categorical(tag)
[x_train, x_test] = np.array_split(x, 2, axis=0)
[y_train, y_test] = np.array_split(tag, 2, axis=0)

pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.99,
                                                   begin_step=100,
                                                   end_step=300,
                                                   frequency=100)
}
from tensorflow.keras.layers import TimeDistributed, Input, Dense, BatchNormalization, Flatten
inputs = Input(shape=(x.shape[1]))
outputs = sparsity.prune_low_magnitude(Dense(2, activation='softmax'), **pruning_params)(inputs)

# node = Dense(76, activation='selu')(inputs)
# node = BatchNormalization()(outputs)
# outputs = Dense(2, activation='softmax')(outputs)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.summary()
ad = tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
logdir = 'prune.log'

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(x_train[:,:], y_train[:,:],
      validation_data=(x_test[:,:],y_test[:,:]),
      batch_size=1,
      epochs=30,      
      callbacks=[  sparsity.UpdatePruningStep(),
                  sparsity.PruningSummaries(log_dir=logdir, profile_batch=0),
                  CSVLogger('pruned_training.csv',append=True),
                  ],
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
