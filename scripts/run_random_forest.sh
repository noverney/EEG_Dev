#!/bin/bash
# for upsampling
for i in {1..100}
do
    #python random_forest_peaks.py --random_state $i --sampling up  --test_size 0.10
    #python random_forest_peaks.py --random_state $i --sampling up  --test_size 0.15
    python random_forest_peaks.py --random_state $i --sampling up  --test_size 0.20
    #python random_forest_peaks.py --random_state $i --sampling up  --test_size 0.25
done

# for downsampling
for i in {1..100}
do
    #python random_forest_peaks.py --random_state $i --sampling down  --test_size 0.10
    #python random_forest_peaks.py --random_state $i --sampling down  --test_size 0.15
    python random_forest_peaks.py --random_state $i --sampling down  --test_size 0.20
    #python random_forest_peaks.py --random_state $i --sampling down  --test_size 0.25
done

#for synthetic 
for i in {1..100}
do
    #python random_forest_peaks.py --random_state $i --sampling syn  --test_size 0.10
    #python random_forest_peaks.py --random_state $i --sampling syn  --test_size 0.15
    python random_forest_peaks.py --random_state $i --sampling syn  --test_size 0.20
    #python random_forest_peaks.py --random_state $i --sampling syn  --test_size 0.25
done

# I am not modifying the number of trees... 
# I am not modifying the maximum depth
