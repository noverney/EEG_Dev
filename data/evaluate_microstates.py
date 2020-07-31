# we will now load the large numpy matrix
import numpy as np
from IPython.core.display import Image
from sef_eeg_microstates import *
import time
import plotly.graph_objects as go
#import chart_studio.plotly as py
import plotly.offline as py
# load the matrix of all the controls
from preprocessing import chan_ID_213 as chs
import itertools
import pandas as pd

# we have to chekc with the t-test but the std are way to high 
# I have to do the welch t-test since the samples are unequal
# http://daniellakens.blogspot.com/2015/01/always-use-welchs-t-test-instead-of.html
from scipy import stats

def load_files(microstates_path, peaks_path):
    microstates = np.load(microstates_path)
    peaks = np.load(peaks_path)
    print("The number of microstates: {0}".format(len(microstates)))
    print("The number of peaks: {0}".format(len(peaks)))
    # we need to split by the sample 
    return microstates, peaks

def split_samples(microstates, peaks, sizes_control, sizes_disease, disease_first=True):
    # we should have a vertical line to draw where the serpation for control vs patients with pd 
    samples = [] 
    # we  need 
    borderline = sum(sizes_control)
    if disease_first:
        borderline = sum(sizes_disease)
    total = sum(sizes_control) + sum(sizes_disease)
    print "The borderline is at: {0}".format(borderline)
    print "The total is at: {0}".format(total)
    print "Length of microstates sequence: {0}".format(len(microstates))
    print len(microstates)
    index = 0
    peak_index = 0
    samples_peaks = []

    combined = sizes_control + sizes_disease
    if disease_first:
        combined = sizes_disease + sizes_control

    for size in combined:
        sample_with_peaks = []
        samples.append(microstates[index:index+size])
        #print peaks[peak_index]
        #print index+size 
        while peak_index < len(peaks) and peaks[peak_index] <= index+size:
            sample_with_peaks.append(microstates[peaks[peak_index]])
            peak_index += 1
        samples_peaks.append(sample_with_peaks)
        index += size
    print peak_index
    print peaks[-1]
    print len(samples_peaks)
    borderline2 = sum([len(x) for x in samples_peaks[:len(sizes_control)]])
    if disease_first:
        borderline2 = sum([len(x) for x in samples_peaks[:len(sizes_disease)]])
    print borderline2

    return samples, borderline, samples_peaks, borderline2 

# finding repeating blocks in only the peaks results 
def find_repeating_blocks(samples_peaks, n_maps):
    microstates = list(itertools.chain.from_iterable(samples_peaks))

    def continual_length(sequence, value):
        lengths = []
        length = 0
        i = 0
        while i < len(sequence):
            if sequence[i] == value:
                length += 1
            else:
                if length > 0:
                    lengths.append((length, i))
                    length = 0
            i += 1
        return lengths

    continual_zero = continual_length(microstates, 0)
    # I can sort these balues but I should maybe start highlighting where they are 
    top_ten = sorted(continual_zero, reverse=True)[:10]
    print top_ten

    # plot the bands on the x axis 
    # time to get all four 
    repeating_blocks = {}
    for i in range(n_maps):
        repeating_blocks[i] = continual_length(microstates, i)
    
    return repeating_blocks

def find_repeating_blocks_start_end(repeating_blocks, n_maps, min_length=5):
    repeat_blocks_start_end = {}

    for i in range(n_maps):
        start = []
        stop = []

        for length, index in repeating_blocks[i]:
            if length >= min_length:
                start.append(index-length)
                stop.append(index)
        repeat_blocks_start_end[i] = {'start': start, 'stop':stop}
    return repeat_blocks_start_end

def find_classes_to_ranges(repeat_blocks_start_end,n_maps, min_range=5):
    # we should calculate how much is within each range intervals 
    # to the greatest length 
    max_range = 0
    class_to_ranges = {}
    for n_class in range(n_maps):
        # y is the ranges 
        class_to_ranges[n_class] = {"x":[], "y":[]}
        
        n = len(repeat_blocks_start_end[n_class]['start'])
        for i in range(n):
            start = repeat_blocks_start_end[n_class]['start'][i]
            stop = repeat_blocks_start_end[n_class]['stop'][i]
            length = stop - start
            
            # we only want to consider the top
            if length < min_range:
                continue
            
            class_to_ranges[n_class]["x"].append(start)
            class_to_ranges[n_class]["y"].append(stop - start)
            
            if length > max_range:
                max_range = length
    print "The maximum sequence: {0}".format(max_range)
    return class_to_ranges, max_range

def subset_repeat_blocks_start_end(repeat_blocks_start_end):
    # we need to filter the segments which are too short to concern us 
    # we can also subset it to certain ranges of interest 
    min_length = 10
    repeat_blocks_start_end_smaller = {}

    for i in range(n_maps):
        start = []
        stop = []

        for length, index in repeating_blocks[i]:
            if length >= min_length:
                start.append(index-length)
                stop.append(index)
        repeat_blocks_start_end_smaller[i] = {'start': start, 'stop':stop}
    return repeat_blocks_start_end_smaller

# If we observe a large p-value, for example larger than 0.05 or 0.1,
# then we cannot reject the null hypothesis of #identical average scores.
#  If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%, 
# then we reject the null hypothesis of equal averages.
def welch_dof(x,y):
    dof = (x.var()/x.size + y.var()/y.size)**2 / ((x.var()/x.size)**2 / (x.size-1) + (y.var()/y.size)**2 / (y.size-1))
    print "Welch-Satterthwaite Degrees of Freedom= {0:.4f}".format(dof)
        
# In statistics, Welch's t-test, or unequal variances t-test, is a two-sample location test which 
# is used to test the hypothesis that two populations have equal means.
# so we set it to false since we assume they do not have equal means 
def welch_ttest(x, y): 
    ## Welch-Satterthwaite Degrees of Freedom ##
    dof = (x.var()/x.size + y.var()/y.size)**2 / ((x.var()/x.size)**2 / (x.size-1) + (y.var()/y.size)**2 / (y.size-1))
   
    t, p = stats.ttest_ind(x, y, equal_var = False)
    
    print "Welch's t-test= {0:.4f}".format(t)
    print "p-value = {0:.4f}".format(p)
    print "Welch-Satterthwaite Degrees of Freedom= {0:.4f}".format(dof)
    return p

# this is more like a util function
# turn dictionary to dataframe
def dict_to_df(samples):
    states = []
    counts = []
    for count in samples:
        states.extend(count.keys())
        counts.extend(count.values())
    return pd.DataFrame({'class':states, 'count':counts})

# make a pie chart for how much between certain ranges of lenght of contigious segments 
# determine the length of streches of 0,1,2,3
# find the max lenghts 
def continual_seq_lengths(sequence, value):
    lengths = []
    length = 0
    i = 0
    while i < len(sequence):
        if sequence[i] == value:
            length += 1
        else:
            if length > 0:
                lengths.append(length)
                length = 0
        i += 1
    return lengths

def get_counts(seq, min_range, n_maps=4):
    counts = {x:[] for x in range(n_maps)}
    for i in range(n_maps):
        lengths = continual_seq_lengths(seq,i)
        counts[i].extend([x for x in lengths if x > min_range])
    return counts

def number_per_sample(sample_to_lengths):
    counts = []
    for sample in sample_to_lengths:
        counts.append({k:len(v) for k,v in sample.iteritems()})
    return counts


if __name__ == "__main__":
    n_maps = 4
    microstates_path = "microstates_Theta_4.npy"
    peaks_path = "peaks_Theta_4.npy"
    microstates, peaks = load_files(microstates_path, peaks_path)
    sizes_control = [712]*24
    sizes_disease = [712]*44

    samples, borderline, samples_peaks, borderline2 = split_samples(microstates, 
                                                                    peaks,
                                                                    sizes_control,
                                                                    sizes_disease)
    repeating_blocks = find_repeating_blocks(samples_peaks, n_maps)
    repeat_blocks_start_end = find_repeating_blocks_start_end(repeating_blocks, n_maps)
    class_to_ranges, max_range = find_classes_to_ranges(repeat_blocks_start_end, n_maps)
    print(len([get_counts(x, 5, n_maps) for x in samples_peaks]))

    np.save("sample_Theta_peaks_4", samples_peaks)