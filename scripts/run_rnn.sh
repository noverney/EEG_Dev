#!/bin/bash
for i in {20..50}
do
    python test_rnn.py data/sample_peaks_4.npy $i 1 100
    python test_rnn.py data/dist_4.npy $i 1 100
    python test_rnn.py data/peaks_with_dist_4.npy $i 1 100
done