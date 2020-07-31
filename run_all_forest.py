from random_forest_peaks import run

seeds=[3,10,20,23,26,33,39,44,51,60,64,75,81]
filepath = "data/sample_peaks_4.npy"

for random_state in seeds:
    for sampling_type in ["up", "down", "syn"]:
        run(filepath, sampling_type, random_state, outpath="data", test_size=0.20, 
            n_maps=4, print_all=False)