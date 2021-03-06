import os
import itertools
# need the train to to return some valus to store in a table 
# loop for the layers

with open('run_lstm.sbatch', 'r') as file:
    batch_file = file.read() 

filepath = "sample_peaks_4.npy"
# possible size of the number of hidden units 
hidden_units = [2048,1024,512,256,128,64]

for layers in range(2,5):
    # get every n C r for the hidden units with repeats 
    combs = list(itertools.combinations(hidden_units,layers))
    for num_hidden in combs:
        num_hidden = list(num_hidden)
        prefix = f"layers{len(num_hidden)}"
        hidden_units_name = ",".join([str(x) for x in num_hidden])
        for random_state in range(10):
            output_name = f"lstm_{hidden_units_name}_{random_state}.txt"
            script = batch_file.format(output_name=output_name, filepath=filepath, 
                                       num_epoches=100, hidden_units=hidden_units_name,
                                       prefix=prefix, random_state=random_state)
            script_name = f"lstm_cuda_{hidden_units_name}_{random_state}.sbatch"
            with open(f"scripts/{script_name}", "w") as file:
                file.write(script)
            os.system(f"chmod 755 ./scripts/{script_name}")
            os.system(f"sbatch ./scripts/{script_name}")