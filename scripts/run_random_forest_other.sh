#!/bin/bash
# for upsampling


seeds=(3 10 20 23 26 33 39 44 51 60 64 75 81)

for j in ${!seeds[@]}
do
for i in {1..10}
do
    python random_forest_peaks.py --random_state $i --sampling up  --test_size 0.25 \
    --filepath "data/sample_peaks_${j}.npy" --n_maps $j;
done

# for downsampling
for i in {1..10}
do
    python random_forest_peaks.py --random_state $i --sampling down  --test_size 0.25 \
     --filepath "data/sample_peaks_${j}.npy" --n_maps $j;
done

#for synthetic 
for i in {1..10}
do
    python random_forest_peaks.py --random_state $i --sampling syn  --test_size 0.25 \
    --filepath "data/sample_peaks_${j}.npy" --n_maps $j;
done
done
# I am not modifying the number of trees... 
# I am not modifying the maximum depth
