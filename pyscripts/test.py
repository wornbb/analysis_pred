import h5py
from pathlib import Path
fname = Path(r'''C:\Users\Yi\Desktop\analysis_pred\pyscripts\balanced_grid_sensor''').joinpath(r"balanced_gird_sensor.blackscholes2c.h5")
with h5py.File(fname, 'r') as f:
    x = f["x"]
    y = f["y"]