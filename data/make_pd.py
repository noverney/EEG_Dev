# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# we will now load the large numpy matrix
import numpy as np
from IPython.core.display import Image
from sef_eeg_microstates import *
import time
import plotly.graph_objects as go
#import chart_studio.plotly as py
import plotly.offline as py
# load the matrix of all the controls
patients_with_pd = np.load('combined_pd_ec.npy')
n_maps = 4
fs = 1000.0

# here we do some clustering
from preprocessing import chan_ID_213 as chs
mode = ["aahc", "kmeans", "kmedoids", "pca", "ica"][1]
print("Clustering algorithm: {:s}".format(mode))
locs = []
maps, x, gfp_peaks, gev = clustering(patients_with_pd, fs, chs, locs, mode, n_maps, doplot=True)
print("The length of the sequence: {0}".format(len(x)))
# maybe just take a subset of the following class 
print len(x)
np.save("microstates_pd",x) # we only need to do this once sine the microstates calculated should be the same