from keras.models import load_model
import pickle
import numpy as np
fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"

[x_train,y_train,x_test,y_test] = pickle.load(open("all_vios.p","rb"))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
x_train = scaler.fit_transform(x_train[:,:,0])
x_train = np.expand_dims(x_train, axis=2)
x_test = scaler.fit_transform(x_test[:,:,0])
x_test = np.expand_dims(x_test, axis=2)


model = load_model('model.06-4.824.hdf5')

scores = model.evaluate(x_test, y_test, verbose=0)
print(sum(y_test)/len(y_train))
print("Accuracy: %.2f%%" % (scores[1]*100))