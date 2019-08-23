from loading import *
f_list = [
"/data/yi/voltVio/analysis/raw/" + "blackscholes2c" + ".gridIR",
"/data/yi/voltVio/analysis/raw/" + "bodytrack2c" + ".gridIR",
"/data/yi/voltVio/analysis/raw/" + "freqmine2c"+ ".gridIR",
"/data/yi/voltVio/analysis/raw/" + "facesim2c"+ ".gridIR",
]
for fname in f_list:
    [a, b] = read_violation(file=fname, trace=10,lines_to_read=10000)
    print(a.shape)