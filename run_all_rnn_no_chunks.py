import itertools
import os
from test_lstm import train
from test_lstm import RNNModelCellsGPU
import time
# need the train to to return some valus to store in a table 
# loop for the layers

total_number = 500
index = 0
filepath = "sample_peaks_4.npy"
start = time.time()

# possible size of the number of hidden units 
hidden_units = [2048,1024,512,256,128,64]
# TODO remove the 4,5 since I have only running the last layer
for layers in range(4,5):
    # get every n C r for the hidden units with repeats 
    combs = list(itertools.combinations(hidden_units,layers))
    for num_hidden in combs:
        num_hidden = list(num_hidden)
        num_hidden_names = ",".join([str(x) for x in num_hidden])
        prefix = f"layers{len(num_hidden)}"
        for random_state in range(10):
            print(f"RNNModelCellsGPU_{num_hidden_names}_{random_state}")
            train(filepath, 50, RNNModelCellsGPU, num_hidden, prefix, random_state)
            index +=1
            print(f"Done with: {index}/{total_number}")
print(f"Total Time: {time.time()-start}")