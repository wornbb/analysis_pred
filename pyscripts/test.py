from loading import *
fname = "F:\\Yaswan2c\\Yaswan2c.gridIR"
[a,b] = read_violation(file=fname, trace=10, lines_to_read=10000)
print(a.shape)