#
# Author: Normand Overney
# 
import numpy as np
from evaluate_microstates import split_samples

# the distances 
# [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 27]
# this function is if I no longe want to use distances but have a range 
# of lets say two but I think that is not needed since not that many 
# more unique classes   
def convert_dist(distances, range_val):
    # e.g. 0 -> 5 class one 5 -> 10 class two....
    distance_to_class = {}
    names = []
    steps = range_val
    class_number = 0
    for i in range(1, max(distances)+1):
        if steps < 0:
            class_number += 1
            steps = range_val
        distance_to_class[i] = class_number
        steps -= 1
    return distance_to_class

def create_distance_classes(peaks, sample_peaks, n_maps = 4):
    distances = []
    for i in range(len(peaks)-1):
        distances.append(peaks[i+1]-peaks[i])
    # print len(distances)
    # print sum([len(x) for x in sample_peaks])
    # the number of uniq distances 
    # print(len(set(distances)))
    # lets add the class to each of them 

    samples_peaks_with_distance = []
    only_distances = []
    index = 0
    for sample in sample_peaks:
        sample_plus_distances = []
        sample_distances = []
        for elem in sample:
            sample_plus_distances.append(elem)
            if index < len(distances):
                sample_plus_distances.append(distances[index] + n_maps)
                sample_distances.append(distances[index])
                index += 1
        samples_peaks_with_distance.append(sample_plus_distances)
        only_distances.append(sample_distances)
    assert sum([len(x) for x in samples_peaks_with_distance]) == \
           len(distances) +  sum([len(x) for x in sample_peaks])

    # at the end of each we need to remove it for every sample except last one
    for index, sample in enumerate(samples_peaks_with_distance[:-1]):
        samples_peaks_with_distance[index] = np.delete(sample, -1)

    assert sum([len(x) for x in samples_peaks_with_distance]) + len(sample_peaks) -1 == \
           len(distances) +  sum([len(x) for x in sample_peaks]) 

    # so we not have the sample with classnames for each of the distances 
    return samples_peaks_with_distance, only_distances


if __name__ == "__main__":
    peaks = np.load("../peaks_4.npy")
    microstates = np.load("../microstates_4.npy")
    print(peaks.shape)
    print(microstates.shape)

    from preprocessing import sizes_control
    from preprocessing import sizes_disease

    # we have to divide everything by ten 
    sizes_control = [x/10 for x in sizes_control]
    sizes_disease = [x/10 for x in sizes_disease]

    samples, borderline, samples_peaks, borderline2 = split_samples(microstates, peaks, sizes_control, sizes_disease, False)
    samples_peaks_with_distance, only_distances = create_distance_classes(peaks, samples_peaks)
    np.save("../peaks_with_dist_4", samples_peaks_with_distance)
    np.save("../dist_4", only_distances)