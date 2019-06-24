import numpy as np
import pickle
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
def preprocess(batch, base_volt, threshold):
    """Find the correct Y given x (batch)
    
    Arguments:
        batch {np 2D} -- loaded voltage on PDN grid
        base_volt {int } -- reference normal voltage
        threshold {int} -- threshold to define violation (in percentage)
    """
    batch = batch / base_volt
    overshoot = batch >= (1 + threshold/100)
    droop = batch <= (1 - threshold/100)
    labeled = overshoot + droop
    batch = (batch - 0.9 * base_volt) * 10

    return batch.astype('float32') , labeled.astype('int')
def prepare_xy(x, labeled, timestep):
    if x.shape[1] % timestep:
        print(x.shape[1] % timestep)
        print('sample/timestep mismatch')
        return 0
    # parameter
    timestep_per_cycle = 5
    input_cycle = 0
    predi_cycle = 0
    inputStep = timestep_per_cycle * input_cycle
    prediStep = timestep_per_cycle * predi_cycle 

    # prepare loop
    startP  = inputStep + prediStep + 1
    half_sample = np.sum(np.any(labeled,0))
    sample_size = 2 * half_sample
    sensor_num = 1
    dim = int(np.sqrt(x.shape[0]))
    X = np.zeros((sample_size, dim, dim, 1))
    Y = np.zeros((sample_size, sensor_num))
    sample_counter = -1
    balance_counter = 0
    for col in range(startP, x.shape[1]):
        if np.any(labeled[:, col]) == 1:
            balance_counter += 1
            sample_counter += 1
            X[sample_counter, :, :, 0] = x[:, col - prediStep].reshape(dim, dim)
            Y[sample_counter, 0] = 1
        else:
            if balance_counter > 0:
                balance_counter -= 1
                sample_counter += 1
                X[sample_counter, :, :, 0] = x[:, col - prediStep].reshape(dim, dim)
                Y[sample_counter, 0] = 0
    print("balance_ounter = ", balance_counter)                 
    return X.astype('float32'), Y.astype('int')

  
with open('saved_data.pk', 'rb') as f:
  save = pickle.load(f)
  
base_volt = 1
threshold = 3
timestep = 50

x_train = save[0]
x_test = save[1]
print(x_train.shape)
x_train, y_train = preprocess(x_train, base_volt, threshold)
x_train, y_train = prepare_xy(x_train, y_train, timestep)
print(y_train.shape)
print(x_train.shape)
x_test, y_test = preprocess(x_test, base_volt, threshold)
x_test, y_test = prepare_xy(x_test, y_test, timestep)

import matplotlib.pyplot as plt
print(y_train[0:10])
plt.plot(x_train[0:10,:,0].reshape(10,76).transpose())
plt.show()

