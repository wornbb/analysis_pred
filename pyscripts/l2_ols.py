import numpy as np
import pickle
from sklearn import linear_model
import matplotlib.pyplot as plt
print("start")
with open('saved_data.pk', 'rb') as f:
  save = pickle.load(f)
  
base_volt = 1
threshold = 2
timestep = 50

x_train = save[0]
x_test = save[1]

reg = linear_model.Ridge(alpha=20)
reg.fit(x_train.transpose()[:-1,:], x_train.transpose()[1:,:]) 
print(reg.coef_.shape)
plt.plot(np.norm(reg.coef_))
plt.show()
print("end")