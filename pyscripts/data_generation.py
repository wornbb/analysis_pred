import pickle
import numpy as np
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
    return batch, labeled
def prepare_xy(x, labeled, timestep):
    if x.shape[1] % timestep:
        print(x.shape[1] % timestep)
        print('sample/timestep mismatch')
        return 0
    # parameter
    timestep_per_cycle = 5
    input_cycle = 2
    predi_cycle = 1
    inputStep = timestep_per_cycle * input_cycle
    prediStep = timestep_per_cycle * predi_cycle 

    # prepare loop
    startP  = inputStep + prediStep + 1
    half_sample = np.sum(np.sum(labeled[startP:,:]))
    sample_size = 2 * half_sample
    sensor_num = 1
    X = np.zeros((sample_size, inputStep, sensor_num))
    Y = np.zeros((sample_size, sensor_num))
    sample_counter = -1
    balance_counter = 0
    for col in range(startP, x.shape[1]):
        for sensor_loc in range(x.shape[0]):
            if labeled[sensor_loc, col] == 1:
                balance_counter += 1
                sample_counter += 1
                X[sample_counter, :, 0] = x[sensor_loc, col - inputStep - prediStep : col - prediStep].transpose()
                Y[sample_counter, 0] = labeled[sensor_loc, col]
            else:
                if balance_counter > 0:
                    balance_counter -= 1
                    sample_counter += 1
                    X[sample_counter, :, 0] = x[sensor_loc, col - inputStep - prediStep : col - prediStep].transpose()
                    Y[sample_counter, 0] = labeled[sensor_loc, col]                    
    return X, Y


with open('saved_data.pk', 'rb') as f:
  save = pickle.load(f)
  
base_volt = 1
threshold = 2
timestep = 50

x_train = save[0]
x_test = save[1]
x_train, y_train = preprocess(x_train, base_volt, threshold)
x_train, y_train = prepare_xy(x_train, y_train, timestep)
print(y_train.shape)
print(x_train.shape)
x_test, y_test = preprocess(x_test, base_volt, threshold)
x_test, y_test = prepare_xy(x_test, y_test, timestep)
with open('data_train.pk', 'wb') as f:
    pickle.dump([x_train, y_train, x_test, y_test], f)
print("Data Generated")
