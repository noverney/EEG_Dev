import itertools
import os
from test_rnn import run
from test_rnn import run_lstm_chunks
from test_rnn import run_plain_rnn
from test_rnn import run_plain_lstm
import time
# need the train to to return some valus to store in a table 
# loop for the layers

total_number = 80
index = 0
filepath = "sample_peaks_4.npy"
start = time.time()
#chunks = [x for x in range(10,16)]
chunks = [15]

# input (batch_size * window_size e.g. 100*10) -> hidden_layers -> output (2 num output classes)
# the numbers represent the number of hidden units so it would 
# 2048*1024 -> 1024*256 -> 256*2
# three_layer = [2048,1024,256]
# two_layer = [2048,512]
# configurations = [two_layer, three_layer]
epoches = 100
#random_states  = [3,10,20,23,26,33,39,44,51,60]
random_states = [x for x in range(1,11)]
hidden_units = [2048,1024,512,256]

configurations = hidden_units
prefix = f"restart"
#configurations.extend(list(itertools.combinations(hidden_units,2)))
#configurations.extend(list(itertools.combinations(hidden_units,3)))

for chunk_size in chunks:
    for layer_dim in [3]:
        for hidden_dim in configurations:
            for random_state in random_states:
                index +=1
                print(f"RNNModelGPU_{hidden_dim}_{layer_dim}_{random_state}")
                run_plain_rnn(filepath, chunk_size,epoches, hidden_dim, layer_dim,True,prefix,random_state)
                print(f"Done with: {index}/{total_number}")
                index +=1
                print(f"LSTMModelGPU_{hidden_dim}_{layer_dim}_{random_state}")
                run_plain_lstm(filepath, chunk_size,epoches, hidden_dim, layer_dim,True,prefix,random_state)
                print(f"Done with: {index}/{total_number}")
print(f"Total Time: {time.time()-start}")

