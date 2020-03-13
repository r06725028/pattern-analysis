import numpy as np
import os, sys, time
from tqdm import tqdm  
import matplotlib.pyplot as plt 

inpath = os.path.join(os.getcwd(),'image')
#outpath = os.path.join(os.getcwd(),'feature')

f2type = {}
data_list = []
for dirpath, dirnames, filenames in os.walk(inpath):
    for f in filenames:
        f2type[f] = dirnames
        data = plt.imread(os.path.join(dirpath, f)) 
        #print(data.shape)
        data = np.mean(data[113:-104,158:-133],2)
        data.resize(128,128)
        #print(data.shape)
        data_list.append(data.reshape(128,128,1))

np.save('data.npy', np.array(data_list).reshape(-1,128,128,1))
np.save('f2type.npy', f2type)  
