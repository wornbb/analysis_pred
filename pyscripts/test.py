import numpy as np
import tables #register blosc
import h5py as h5
import h5py_cache as h5c
import time

batch_size=120
train_shape=(90827, 10, 10, 2048)
hdf5_path='Test.h5'
# As we are writing whole chunks here this isn't realy needed,
# if you forget to set a large enough chunk-cache-size when not writing or reading 
# whole chunks, the performance will be extremely bad. (chunks can only be read or written as a whole)
f = h5c.File(hdf5_path, 'w',chunk_cache_mem_size=1024**2*200) #200 MB cache size
dset_train_bottle = f.create_dataset("train_bottle", shape=train_shape,dtype=np.float32,chunks=(10, 10, 10, 2048),compression=32001,compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
prediction=np.array(np.arange(120*10*10*2048),np.float32).reshape(120,10,10,2048)
t1=time.time()
#Testing with 2GB of data
for i in range(20):
    #prediction=np.array(np.arange(120*10*10*2048),np.float32).reshape(120,10,10,2048)
    #dset_train_bottle[i*batch_size:(i+1)*batch_size,:,:,:]=prediction
    np.save('kkk',prediction,allow_pickle=False)

f.close()
print(time.time()-t1)
print("MB/s: " + str(2000/(time.time()-t1)))