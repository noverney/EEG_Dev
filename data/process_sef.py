from IPython.core.display import Image
from sef_eeg_microstates import *
from preprocessing import read_sef
import time 
from os import listdir, path
import time
from sys import getsizeof


def get_data(filepath):
    chs, fs, data_raw = read_sef(filepath)
    data_raw = np.array(data_raw[:,:213])
    # I am dropping the REF from the dataset because I have no idea what it is 
    # print data_raw.shape
    return data_raw

#
# This function is obsoleted 
# 
def get_all_files(directory, save_filename):
    files = listdir(directory)
    # check they are all sef files
    sef_files = [x for x in files if x[-4:] == ".sef"]
    matrix_shape = (0,213)
    for filename in sef_files:
        matrix_shape = (matrix_shape[0] + get_data(path.join(directory, filename)).shape[0] , 213)
    # now let us see if I can allocate it 
    test = np.ndarray(shape=matrix_shape, dtype=float, order='F')
    print test.nbytes / 1024 / 1024
    print matrix_shape
    size = 0
    for filename in sef_files:
        temp = get_data(path.join(directory, filename))
        #print test[size:temp.shape[0],:].shape
        print temp.shape
        test[size:size+temp.shape[0],:] = temp
        size += temp.shape[0]
        del temp
    #print test[0:10,:]
    # save it to a file then load it up
    np.save(save_filename, test)

def get_size(directory, every_other=1):
    start = time.time()
    files = listdir(directory)
    # check they are all sef files
    sef_files = [x for x in files if x[-4:] == ".sef"]
    columns = 0
    sizes = []
    fullpaths = []
    for filename in sef_files:
        fullpath = path.join(directory, filename)
        size = int(get_data(fullpath).shape[0] / every_other)
        sizes.append(size)
        columns += size
        fullpaths.append(fullpath)
    print "Done with {0} files in {1}".format(len(sef_files), directory)
    print "Time took: {0}".format(time.time() - start)
    return columns, fullpaths, sizes

# Patient CT037 and Patient CT020 data seemed to have problems
# I can already just get a tenth of everyhting here if it does not 
# work 
def get_all_data(control_directory, disease_directory, save_filename, every_other=1):
    columns_control, sef_files_control, sizes_control = get_size(control_directory, every_other)
    columns_disease, sef_files_disease, sizes_disease = get_size(disease_directory, every_other)
    
    sef_files = sef_files_control + sef_files_disease
    matrix_shape = (columns_control+columns_disease, 213)
    print matrix_shape
    # allocate matrix
    test = np.ndarray(shape=matrix_shape, dtype=float, order='F')
    print test.nbytes / 1024 / 1024

    size = 0
    for count, filepath in enumerate(sef_files):
        if every_other > 1:
            temp = get_data(filepath)[0::every_other,:]
        else:
            temp = get_data(filepath)
        #print temp.shape
        print "{0}/{1} Finished with {2}".format(count,len(sef_files), filepath)
        test[size:size+temp.shape[0],:] = temp
        size += temp.shape[0]
        del temp
    # save it to a file then load it up
    print sizes_control
    print sizes_disease
    np.save(save_filename, test)

if __name__ == "__main__":
    start = time.time()
    control_directory = "/home/rehaxu22/Documents/pd_data/band1_35Hz_HC_EC"
    disease_directory = "/home/rehaxu22/Documents/pd_data/band1_35Hz_PD_EC"
    #directory = "D:/pd_data/band1_35Hz_HC_EC"
    #directory = "D:/pd_data/band1_35Hz_PD_EC"
    
    filename = "combined_all_data"
    # just took a short 17 minutes 
    #get_all_data(control_directory, disease_directory, filename)

    x,y, disease_sizes = get_size(control_directory)
    x,y, control_sizes = get_size(disease_directory)
    print disease_sizes
    print control_sizes

    # here we have a list of the sizes so we can part it out 
    

    print "total time: " + str(time.time() - start)

