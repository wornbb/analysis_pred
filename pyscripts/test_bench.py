from keras.models import load_model
import pickle
import numpy as np
import h5py

fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"
save_fname = "combined_lstm_training.data"
with h5py.File(save_fname,"r") as hf:
        x = hf["x"].value
        y= hf["y"].value
# dirty fixing
x = np.squeeze(x, axis=(0,2))
# x = np.squeeze(x, axis=(2))
y = y.flatten().astype('int')
shuffle_index = np.arange(y.shape[0])
np.random.shuffle(shuffle_index)
x = x[shuffle_index,:]
y = y[shuffle_index]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
pred_str = 5
scaled_x = scaler.fit_transform(x[:,:-pred_str])
scaled_x = np.expand_dims(scaled_x, axis=2)
train_size = x.shape[0]//4
x_train = scaled_x[:train_size,:]
x_test = scaled_x[train_size:,:]

y_train = y[:train_size]
y_test = y[train_size:]

 
#model = load_model('31.model.23-0.722.hdf5')
#model = load_model('nn.32.model.12-0.684.hdf5')
#model = load_model('nn.32.model.15-0.605.hdf5')
#model = load_model('nn.34.biLSTM.18-0.171.hdf5') #96.45
model = load_model('residual.3.biLSTM.45.15-0.997-0.008.hdf5')
print(model.metrics_names)
scores = model.evaluate(scaled_x[:,:34,:], y, verbose=0)
print(sum(y_test)/len(y_test))
print("Accuracy: %.2f%%" % (scores[1]*100))