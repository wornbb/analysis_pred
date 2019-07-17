import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from loading import generate_prediction_data
from tensorflow.keras.utils import to_categorical
# [data, tag] = pickle.load(open('balanced_grid_sensor.data', 'rb'))
# tag = to_categorical(tag)
# [x_train, x_test] = np.array_split(data, 2, axis=1)
# [y_train, y_test] = np.array_split(tag, 2, axis=1)
from tensorflow.keras.layers import TimeDistributed, Input, Dense, BatchNormalization
sensor_model = load_model('nn.26.biLSTM.32.14-0.381.hdf5')


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
node = sparsity.prune_low_magnitude((Dense(x_train.shape[0], activation='selu'), **pruning_params)(node)
node = BatchNormalization()(node)
output = Dense(2, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(node)
model = tf.keras.models.Model(inputs=x, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.summary()
model.fit(x_train, y_train,
      batch_size=5,
      validation_data=(x_test,y_test),
      epochs=25,
      verbose=1)
