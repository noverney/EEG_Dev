import itertools
import os
from test_rnn import run
import time
# need the train to to return some valus to store in a table 
# loop for the layers

total_number = 270
index = 0
filepath = "peaks_with_dist_4.npy"
start = time.time()

# input (batch_size * window_size e.g. 100*10) -> hidden_layers -> output (2 num output classes)
# the numbers represent the number of hidden units so it would 
# 2048*1024 -> 1024*256 -> 256*2
# three_layer = [2048,1024,256]
# two_layer = [2048,512]
# configurations = [two_layer, three_layer]
epoches = 100
random_states  = [3,10,20,23,26,33,39,44,51,60]
hidden_units = [2048,1024,512,256,128,64]

config = [2048,1024,512]
config2 = [2048,512]
config3 = [1024]

configs = [config, config2, config3]
prefix = "test_chunks"

for chunk_size in range(10,100,10):
    for hidden_dims in configs:
        hidden_dims_name = ",".join([str(x) for x in hidden_dims])
        for random_state in range(10):
            print(f"RNNModelMultiLayer_{hidden_dims_name}_{chunk_size}_{random_state}")
            run(filepath, chunk_size,epoches, hidden_dims,False,prefix,random_state)
            index +=1
            print(f"Done with: {index}/{total_number}")
print(f"Total Time: {time.time()-start}")
