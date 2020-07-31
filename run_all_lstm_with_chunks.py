import itertools
import os
from test_rnn import run
from test_rnn import run_lstm_chunks
import time
# need the train to to return some valus to store in a table 
# loop for the layers

total_number = 2000
index = 0
filepath = "sample_peaks_4.npy"
start = time.time()
chunks = [x for x in range(10,16)]

# possible size of the number of hidden units 
hidden_units = [2048,1024,512,256,128,64]
for chunk_size in chunks:
    for layers in range(2,5):
        # get every n C r for the hidden units with repeats 
        combs = list(itertools.combinations(hidden_units,layers))
        for num_hidden in combs:
            num_hidden = list(num_hidden)
            num_hidden_names = ",".join([str(x) for x in num_hidden])
            prefix = f"layers{len(num_hidden)}"
            for random_state in range(10):
                print(f"LSTMModelCellsGPU_{num_hidden_names}_{random_state}")
                run_lstm_chunks(filepath, chunk_size,50, num_hidden,True,prefix,random_state)
                index +=1
                print(f"Done with: {index}/{total_number}")
print(f"Total Time: {time.time()-start}")
