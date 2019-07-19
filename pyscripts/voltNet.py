import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py
import numpy as np
from loading import generate_prediction_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

with h5py.File('./balanced_grid_sensor.h5','r') as f:
      data = f["data"].value
      tag = f["tag"].value
tag = to_categorical(tag)
[x_train, x_test] = np.array_split(data, 2, axis=0)
[y_train, y_test] = np.array_split(tag, 2, axis=0)
from tensorflow.keras.layers import TimeDistributed, Input, Dense, BatchNormalization, Flatten
sensor_model = load_model('selu.28.biLSTM32.12-0.873.hdf5')
for layer in sensor_model.layers:
      layer.trainable = False
sensor_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
from tensorflow_model_optimization.sparsity import keras as sparsity
end_step = 2000*100
pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=2000,
                                                   end_step=end_step,
                                                   frequency=100)
}

x = Input(shape=(x_train.shape[1], x_train.shape[2], 1))

node = TimeDistributed(sensor_model)(x)
node = Flatten()(node)
node = sparsity.prune_low_magnitude(Dense(1024, activation='relu'), **pruning_params)(node)
node = BatchNormalization()(node)
node = Dense(512, activation="selu")(node)
output = Dense(2, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(node)
model = tf.keras.models.Model(inputs=x, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.summary()


logdir = 'prune.log'
filepath = "pruned" + ".{epoch:02d}-{val_loss:.3f}.hdf5"
model.fit(x_train, y_train,
      batch_size=5,
      validation_data=(x_test,y_test),
      epochs=25,
      callbacks=[  sparsity.UpdatePruningStep(),
                  sparsity.PruningSummaries(log_dir=logdir, profile_batch=0),
                  CSVLogger('pruned_training.csv',append=True),
                  ModelCheckpoint(filepath, monitor='val_categorical_accuracy',save_best_only=True, verbose=1, mode='max')
                  ],
      verbose=1)
