import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py
import numpy as np
from loading import generate_prediction_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from clr_callback import*
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import TimeDistributed, Input, Dense, BatchNormalization, Flatten
from loading import *
from tensorflow_model_optimization.sparsity import keras as sparsity
fname = "./balanced_gird_sensor.facesim2c.h5"
[x_train, y_train, x_test, y_test] = load_h5_tag_grid(fname)
model_name = 'residual.3.biLSTM.45.15-0.997-0.008.hdf5'
sensor_model = load_frozen_lstm(model_name)
end_step = 3500
pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=500,
                                                   end_step=1500,
                                                   frequency=100)
}

print("Input Sequency Lenght: ", x_train.shape[2])
x = Input(shape=(x_train.shape[1],x_train.shape[2],1))
node = TimeDistributed(sensor_model)(x)
node = Flatten()(node)
node = sparsity.prune_low_magnitude(Dense(2, activation='softmax'), **pruning_params)(node)
# node = Dense(76, activation='selu', kernel_initializer=selu_ini)(node)
# node = BatchNormalization()(node)
# output = Dense(2, activation='softmax')(node)
model = tf.keras.models.Model(inputs=x, outputs=node)
ad = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.summary()

logdir = 'prune.log'
filepath = "pruned" + ".{epoch:02d}-{val_categorical_accuracy:.3f}-{val_loss:.3f}.hdf5"
model.fit(x_train[:,:,:,:], y_train[:,:],
      batch_size=3,
      validation_data=(x_test[::10,:,:,:],y_test[::10,:]),
      epochs=2000,
      callbacks=[  sparsity.UpdatePruningStep(),
                  sparsity.PruningSummaries(log_dir=logdir, profile_batch=0),
                  CSVLogger('pruned_training.csv',append=True),
                  ModelCheckpoint(filepath, monitor='val_categorical_accuracy',save_best_only=True, verbose=1, mode='max'),
                  ],
      verbose=1)

                  # CyclicLR(base_lr=0.05, max_lr=0.15, mode='triangular2')