from sef import read
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

# these are hardcoded constant 
chan_ID_256 =  [ 'E1',   'F8',   'E3',   'E4',   'F2',  'E6',    'E7',   'E8',   'E9',  'AF8',  'E11',  'AF4',  'E13',
                'E14',  'FCz',  'E16',  'E17',  'FP2',  'E19',  'E20',   'Fz',  'E22',  'E23',  'FC1',  'E25',  'FPz',
                'E27',  'E28',   'F1',  'E30',  'E31',  'E32',  'E33',  'AF3',  'E35',   'F3',  'FP1',  'E38',  'E39',
                'E40',  'E41',  'FC3',  'E43',   'C1',  'E45',  'AF7',   'F7',   'F5',  'FC5',  'E50',  'E51',  'E52',
                'E53',  'E54',  'E55',  'E56',  'E57',  'E58',   'C3',  'E60',  'E61',  'FT7',  'E63',   'C5',  'E65',
                'CP3',  'FT9',   'T9',   'T7',  'E70',  'E71',  'E72',  'E73',  'E74',  'E75',  'CP5',  'E77',  'E78',
                'CP1',  'E80',  'E81',  'E82',  'E83',  'TP7',  'E85',   'P5',   'P3',   'P1',  'E89',  'CPz',  'E91',
                'E92',  'E93',  'TP9',  'E95',   'P7',  'PO7',  'E98',  'E99', 'E100',   'Pz', 'E102', 'E103', 'E104',
               'E105',   'P9', 'E107', 'E108',  'PO3', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115',   'O1', 'E117',
               'E118',  'POz', 'E120', 'E121', 'E122', 'E123', 'E124', 'E125',   'Oz', 'E127', 'E128', 'E129', 'E130',
               'E131', 'E132', 'E133', 'E134', 'E135', 'E136', 'E137', 'E138', 'E139',  'PO4', 'E141',   'P2',  'CP2',
               'E144', 'E145', 'E146', 'E147', 'E148', 'E149',   'O2', 'E151', 'E152',   'P4', 'E154', 'E155', 'E156',
               'E157', 'E158', 'E159', 'E160',  'PO8',   'P6', 'E163',  'CP4', 'E165', 'E166', 'E167', 'E168',  'P10',
                 'P8', 'E171',  'CP6', 'E173', 'E174', 'E175', 'E176', 'E177', 'E178',  'TP8', 'E180', 'E181', 'E182',
                 'C4', 'E184',   'C2', 'E186', 'E187', 'E188', 'E189', 'TP10', 'E191', 'E192', 'E193',   'C6', 'E195',
               'E196', 'E197', 'E198', 'E199', 'E200', 'E201',   'T8', 'E203', 'E204', 'E205',  'FC4',  'FC2', 'E208',
               'E209',  'T10',  'FT8', 'E212',  'FC6', 'E214', 'E215', 'E216', 'E217', 'E218', 'FT10', 'E220', 'E221',
                 'F6', 'E223',   'F4', 'E225',  'F10', 'E227', 'E228', 'E229', 'E230', 'E231', 'E232', 'E233', 'E234',
               'E235', 'E236', 'E237', 'E238', 'E239', 'E240', 'E241', 'E242', 'E243', 'E244', 'E245', 'E246', 'E247',
               'E248', 'E249', 'E250', 'E251',   'F9', 'E253', 'E254', 'E255', 'E256' ]

chan_ID_213 = [  'E1',   'F8',   'E3',   'E4',   'F2',   'E6',   'E7',   'E8',   'E9',  'AF8',  'E11',  'AF4',  'E13',
                'E14',  'FCz',  'E16',  'E17',  'FP2',  'E19',  'E20',   'Fz',  'E22',  'E23',  'FC1',  'E25',  'FPz',
                'E27',  'E28',   'F1',  'E30',  'E32',  'E33',  'AF3',  'E35',   'F3',  'FP1',  'E38',  'E39',  'E40',
                'E41',  'FC3',  'E43',   'C1',  'E45',  'AF7',   'F7',   'F5',  'FC5',  'E50',  'E51',  'E52',  'E53',
                'E54',  'E55',  'E56',  'E57',  'E58',   'C3',  'E60',  'E61',  'FT7',  'E63',   'C5',  'E65',  'CP3',
                'FT9',   'T9',   'T7',  'E70',  'E71',  'E72',  'E73',  'E74',  'E75',  'CP5',  'E77',  'E78',  'CP1',
                'E80',  'E81',  'E83',  'TP7',  'E85',   'P5',   'P3',   'P1',  'E89',  'CPz',  'E93',  'TP9',  'E95',
                 'P7',  'PO7',  'E98',  'E99', 'E100',   'Pz', 'E103', 'E104', 'E105',   'P9', 'E107', 'E108',  'PO3',
               'E110', 'E112', 'E113', 'E114', 'E115',   'O1', 'E117', 'E118',  'POz', 'E121', 'E122', 'E123', 'E124',
               'E125',   'Oz', 'E127', 'E128', 'E129', 'E130', 'E131', 'E132', 'E134', 'E135', 'E136', 'E137', 'E138',
               'E139',  'PO4', 'E141',   'P2',  'CP2', 'E144', 'E146', 'E147', 'E148', 'E149',   'O2', 'E151', 'E152',
                 'P4', 'E154', 'E155', 'E156', 'E157', 'E158', 'E159', 'E160',  'PO8',   'P6', 'E163',  'CP4', 'E166',
               'E167', 'E168',  'P10',   'P8', 'E171',  'CP6', 'E173', 'E175', 'E176', 'E177', 'E178',  'TP8', 'E180',
               'E181', 'E182',   'C4', 'E184',   'C2', 'E186', 'E188', 'E189', 'TP10', 'E191', 'E192', 'E193',   'C6',
               'E195', 'E196', 'E197', 'E198', 'E200', 'E201',   'T8', 'E203', 'E204', 'E205',  'FC4',  'FC2',  'T10',
                'FT8', 'E212',  'FC6', 'E214', 'E215', 'E218', 'FT10', 'E220', 'E221',   'F6', 'E223',   'F4', 'E225',
                'F10', 'E227',   'F9', 'E253', 'E254']  # subset of 213 electrodes

chan_ID_64 = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 
              'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4',
              'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 
              'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 
              'F2', 'AF4', 'AF8', 'VEOG']

best_channels = ['F3', 'Pz', 'P3', 'P7', 'Oz', 'CP6', 'T8', 'FC6', 'F4', 'AF3', 'P1', 'PO7', 'PO8', 'FC4', 'F6', 'TP8']

# in order of the combined 
# for control all the same size at 177900
sizes_control = [177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 
                  177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900]

# for disease we have a few files which are 176400 long and there are only two of them 
# so missing only 1500 time steps which is just 1.5 seconds 
sizes_disease = [177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 
                 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 
                 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 
                 177900, 176400, 177900, 176400, 177900]

def read_all_sef(directory):
    filenames = []
    for f1 in os.listdir(directory):
        f2 = os.path.join(directory, f1)
        if os.path.isfile(f2):
            if (f2[-4:] == ".sef"):
                filenames.append(f2)
    print(filenames)
    _, _, result = read_sef(filenames[0])
    print(result.shape)
    for index, filename in enumerate(filenames[1:]):
        _, _, temp = read_sef(filename)
        print(temp.shape)
        result = np.append(result, temp, axis=0) 
        print(result.shape)
        print("Done:{0}/{1}".format(index+2, len(filenames)))
    print(result.shape)
    return result

def read_sef(filename):
    infoArray,channelArray,dataArray = read(filename)

    size = infoArray[3][1]
    # we want to print out the channel area shape 
    # so once we have the dataArray we need it to read in into 
    # Wegner's script 
    # it seems to be in the decimal places and the data array shape 
    # is (241000, 214) compared to the (48000, 30) so a lot more data 
    # and more channels 
    return chan_ID_213, 1000.0, dataArray

def generate_xyz(write=False):
    electodes = "electrodes257_eegmachine.csv"
    df = pd.read_csv(electodes, sep='\t', header=None)
    electrodes256 = df[:256]
    # now we add the column for the 
    # we want to overwrite the original label column since it does not 
    # have any of the meaningful names 
    electrodes256[3] = chan_ID_256
    #print electrodes256.shape
    #print electrodes256[:1]

    # we want to get only the part of the electrodes
    indices = [elem in chan_ID_213 for elem in electrodes256[3]]
    electrodes213 = electrodes256[indices]
    #print electrodes213.shape

    # now we add the header after rearranging the columns 
    final_data = {"Site":electrodes213[3], "x":electrodes213[0], "y":electrodes213[1], 
                  "z":electrodes213[2]}
    result = pd.DataFrame(data=final_data)

    # we write it to tsv file 
    if write:
        result.to_csv("bigger_cap.xyz", sep="\t", header=True, index=False)
    return result


def plot_mind_map():
    electodes = "2d_electrodes257.csv"
    df = pd.read_csv(electodes, sep=',', header=None)
    electrodes256 = df[:256]
    # now we add the column for the 
    # we want to overwrite the original label column since it does not 
    # have any of the meaningful names 
    electrodes256[2] = chan_ID_256
    print electrodes256.shape
    #print electrodes256[:1]

    # we want to get only the part of the electrodes
    indices = [elem in chan_ID_213 for elem in electrodes256[2]]
    electrodes213 = electrodes256[indices]
    print electrodes213.shape

    # now we add the header after rearranging the columns 
    final_data = {"Site":electrodes213[2], "x":electrodes213[0], "y":electrodes213[1]}

    y = electrodes213[0].tolist()
    z = electrodes213[1].tolist()
    n = electrodes213[2].tolist()

    fig, ax = plt.subplots()
    ax.scatter(z, y)

    for i, txt in enumerate(n):
        ax.annotate(txt, (z[i], y[i]))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

if __name__ == "__main__":
    #plot_mind_map()
    np.save("D:\\pd_data\\patients", read_all_sef("D:\\pd_data\\band1_35Hz_PD_EC"))
    np.save("D:\\pd_data\\controls", read_all_sef("D:\\pd_data\\band1_35Hz_HC_EC"))

    