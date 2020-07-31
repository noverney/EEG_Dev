import itertools
import os
from test_seq import run
import time
# need the train to to return some valus to store in a table 
# loop for the layers

total_number = 25
index = 0
filepath = "sample_peaks_4.npy"
start = time.time()
# possible size of the number of hidden units 
hidden_units = [128,64,32,16,8]
for layers in range(1,4):
    # get every n C r for the hidden units with repeats 
    combs = list(itertools.combinations(hidden_units,layers))
    for num_hidden in combs:
        num_hidden = list(num_hidden)
        num_hidden_names = ",".join([str(x) for x in num_hidden])
        prefix = f"layers{len(num_hidden)}"
        print(f"SequentialModel_{num_hidden_names}")
        run(filepath, prefix, num_hidden, num_epochs=2000, num_random=10)
        index +=1
        print(f"Done with: {index}/{total_number}")
print(f"Total Time: {time.time()-start}")
