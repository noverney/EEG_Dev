#
# Author: Normand Overney
#

from data.sef import read
import os
from sklearn import preprocessing
import time
import numpy as np


fs = 1000 # Sampling rate (1000 Hz)

def get_eeg_bands(channel_data, display=False):
    band_to_data = {}
    # Get real amplitudes of FFT (only in postive frequencies)
    fft_vals = np.absolute(np.fft.rfft(channel_data))

    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(len(channel_data), 1.0/fs)

    # Define EEG bands
    eeg_bands = {'Delta': (0, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}

    my_colors = ["g", "b","r","y","k"]

    # Take the mean of the fft amplitude for each EEG band
    eeg_band_fft = dict()
    for band in eeg_bands:  
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                           (fft_freq <= eeg_bands[band][1]))[0]
        eeg_band_fft[band] = np.mean(fft_vals[freq_ix])
        band_to_data[band] = channel_data[freq_ix]
    
    if display:
        # Plot the data (using pandas here cause it's easy)
        df = pd.DataFrame(columns=['band', 'val'])
        df['band'] = eeg_bands.keys()
        df['val'] = [eeg_band_fft[band] for band in eeg_bands]
        ax = df.plot.bar(x='band', y='val', legend=False, color=my_colors)
        ax.set_xlabel("EEG band")
        ax.set_ylabel("Mean band Amplitude")
    
    return band_to_data


def get_bands_to_matrix(sample_data):
    data = np.array(sample_data)
    bands_to_matrix = {}
    for channel_index in range(data.shape[1]):
        band_to_data = get_eeg_bands(data[:,channel_index])
        for band in band_to_data:
            channel_data = np.array(band_to_data[band])
            channel_data = np.reshape(channel_data, (channel_data.shape[0],-1))

            if band not in bands_to_matrix:
                bands_to_matrix[band] = channel_data
            else:
                bands_to_matrix[band] = np.append(bands_to_matrix[band], channel_data, axis=1)
    return bands_to_matrix

def get_all_raw(dir_path):
    filepaths = [os.path.join(dir_path,x) for x in os.listdir(dir_path) if ".sef" == os.path.splitext(x)[1]]
    band_to_data = {}
    for index,filepath in enumerate(filepaths):
        sample_data = read(filepath)[2]
        bands_to_matrix = get_bands_to_matrix(sample_data)
        for band in bands_to_matrix:
            band_data = bands_to_matrix[band]
            norm_data = preprocessing.scale(sample_data)
            if band not in band_to_data:
                band_to_data[band]= [np.array(norm_data)]
            else:
                band_to_data[band].append(np.array(norm_data))
        print(f"Done with: {index+1}/{len(filepaths)}, Length: {sample_data.shape[0]}")
    return band_to_data

def zero_fill(X):
    max_length = 0
    for sample in X:
        if sample.shape[0] > max_length:
            max_length = sample.shape[0]
    print(f"Max Length: {max_length}")

if __name__ == "__main__":
    start = time.time()
    pd_dir = "D:/pd_data/band1_35Hz_PD_EC"
    hc_dir = "D:/pd_data/band1_35Hz_HC_EC" 
    band = "Alpha"
    pd_data = get_all_raw(pd_dir)
    hc_data = get_all_raw(hc_dir)
    for band in pd_data:
        X = pd_data[band] + hc_data[band]
        np.save(f"samples_{band}", X)
    print(f"Total time: {time.time()-start}")
    

