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
tag = to_categorical(tag)
[x_train, x_test] = np.array_split(data[:,:,6:-5,:], 2, axis=0)
[y_train, y_test] = np.array_split(tag, 2, axis=0)
from tensorflow.keras.layers import TimeDistributed, Input, Dense, BatchNormalization, Flatten
sensor_model = load_model('selu.28.biLSTM32.12-0.873.hdf5')
for layer in sensor_model.layers:
      layer.trainable = False
sensor_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
from tensorflow_model_optimization.sparsity import keras as sparsity
end_step = 3500
pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=2000,
                                                   end_step=end_step,
                                                   frequency=100)
}

#x = Input(shape=(x_train.shape[1], x_train.shape[2], 1))
#x = Input(shape=(x_train.shape[1],34,1))
x = Input(shape=(5776))
node = BatchNormalization()(x)
#node = TimeDistributed(sensor_model)(x)
node = Flatten()(node)
selu_ini = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1/40, seed=None)
#node = sparsity.prune_low_magnitude(Dense(76, activation='selu', kernel_initializer=selu_ini), **pruning_params)(node)
node = Dense(76, activation='selu', kernel_initializer=selu_ini)(node)
node = BatchNormalization()(node)
node = Dense(10, activation='selu', kernel_initializer=selu_ini)(node)
node = BatchNormalization()(node)
#node = Dense(50, activation="selu", kernel_initializer=selu_ini)(node)
output = Dense(2, activation='softmax')(node)
model = tf.keras.models.Model(inputs=x, outputs=output)
ad = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=ad, metrics=['categorical_accuracy'])
model.summary()

x_train = np.random.rand(y_train.shape[0],5776)
logdir = 'prune.log'
filepath = "pruned" + ".{epoch:02d}-{val_categorical_accuracy:.3f}-{val_loss:.3f}.hdf5"
# model.fit(x_train[::31,:,:34,:], y_train[::31,:],
#       batch_size=3,
#       validation_data=(x_test[::91,:,:34,:],y_test[::91,:]),
#       epochs=2000,
#       callbacks=[  sparsity.UpdatePruningStep(),
#                   sparsity.PruningSummaries(log_dir=logdir, profile_batch=0),
#                   CSVLogger('pruned_training.csv',append=True),
#                   ModelCheckpoint(filepath, monitor='val_categorical_accuracy',save_best_only=True, verbose=1, mode='max'),
#                   ],
#       verbose=1)

                  # CyclicLR(base_lr=0.05, max_lr=0.15, mode='triangular2')
model.fit(x_train[::31,:], y_train[::31,:],
      batch_size=3,
      epochs=2000,
      verbose=1)