from loading import *
import matplotlib.pyplot as plt
import h5py

lstm = load_frozen_lstm("voltnet.lstm.str.5.len50.hdf5")
with h5py.File("F:\\facesim2c.lstm.str5.scaled", 'r') as f:
    x = f["x"][:20000,...]
    y = f["y"][:20000,...]
lstm.evaluate(x,y)
