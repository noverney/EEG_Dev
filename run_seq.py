#
# Author: Normand Overney
#
import time
from test_seq import train
import itertools
import ntpath

filepath = "data/sample_peaks_4.npy"
num_epochs = 100
n_maps = int(ntpath.basename(filepath).split(".npy")[0].split("_")[-1])
hidden_units = [64,32,16,8]
for layers in range(1,4):
    combs = list(itertools.combinations(hidden_units,layers))
    for num_hidden in combs:
        start = time.time()
        hidden_states = list(num_hidden)
        print(hidden_states)
        prefix = f"layers{len(hidden_states)}"
        avg_acc = 0
        num_random = 100
        total_number = num_random
        total_too_fast = 0
        for random_state in range(num_random):
            #print(random_state)
            top_acc, too_fast = train(filepath, n_maps, num_epochs, hidden_states, random_state, prefix, print_info=False)
            if not top_acc:
                #print("Invalid Testing Set")
                total_number -= 1 
            else:
                #print(f"Total Accuracy: {top_acc}, Fast: {too_fast}")
                if too_fast:
                    total_too_fast += 1
                avg_acc += top_acc
        print(f"Average Accuracy: {avg_acc/total_number}")
        print(f"Number of Too fast: {total_too_fast}/{total_number}")
        print(f"Total Time: {time.time()-start}")
