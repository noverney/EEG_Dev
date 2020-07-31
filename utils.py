# 
# Author: Normand Overney
#
# Description: this is for python 3.7 pytorch to help create better samples
#  
#

# in order to generate synthetic samples
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss 
import numpy as np
import matplotlib.pyplot as plt
import random
import os.path
from os import path
from matplotlib.ticker import MultipleLocator
# imbalance 
from sklearn.metrics import balanced_accuracy_score
from collections import Counter

# to split based on lengths 
from itertools import islice 

# these are too be hard coded 
sizes_control = [17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 
                 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 
                 17790, 17790, 17790, 17790]

# for disease we have a few files which are 176400 long and there are only two of them 
# so missing only 1500 time steps which is just 1.5 seconds 
sizes_disease = [17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 
                 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790,
                 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 
                 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 17790, 
                 17640, 17790, 17640, 17790]

def print_class_number(Y):
    class_one = [x for x in Y if x == 0]
    class_two = [x for x in Y if x == 1]
    
    print(f"Number of class one: {len(class_one)}")
    print(f"Number of class two: {len(class_two)}")
    return len(class_one), len(class_two)

def get_class_number(Y):
    return len([x for x in Y if x == 0]), len([x for x in Y if x == 1])

def generate_synthetic_samples(X, Y, print_all=True, random_state=42):
    #print(len(X))
    #print(len(Y))
    if print_all:
        print_class_number(Y)
    sm = SMOTE(random_state=random_state)#, ratio=1.0)
    return sm.fit_sample(X, Y)

def generate_synthetic_samples_alt(X, Y, print_all=True, version=1):
    if print_all:
        print_class_number(Y)
    nr = NearMiss(version=version)
    return nr.fit_sample(X,Y)

def upsample(x,y, seed):
    np.random.seed(seed)
    # Indicies of each class' observations
    i_class0 = np.where(y == 0)[0]
    i_class1 = np.where(y == 1)[0]

    # Number of observations in each class
    n_class0 = len(i_class0)
    n_class1 = len(i_class1)

    # For every observation in class 1, randomly sample from class 0 with replacement
    i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)

    # Join together class 0's upsampled target vector with class 1's target vector
    y = np.concatenate((y[i_class0_upsampled], y[i_class1]))
    x = np.concatenate((x[i_class0_upsampled], x[i_class1]))
    return x,y
    

# this is only for two classes really dumb 
def analyze_chunks(chunks, labels):
    unique_chunks = []
    unique_labels = []

    unique_chunks_set_class_one = set()
    unique_chunks_set_class_two = set()
    not_diff = 0

    # for one class we just add to the set 
    for chunk,label in zip(chunks,labels):
        test = "".join([str(x) for x in chunk])
        if label == 0:
            unique_chunks_set_class_one.add(test)
        else:
            unique_chunks_set_class_two.add(test)

    for chunk, label in zip(chunks,labels):
        test = "".join([str(x) for x in chunk])
        if label == 0:
            if test not in unique_chunks_set_class_two:
                unique_chunks.append(chunk)
                unique_labels.append(label)
            else:
                not_diff += 1
        else:
            if test not in unique_chunks_set_class_one:
                unique_chunks.append(chunk)
                unique_labels.append(label)
            else:
                not_diff += 1
    

    print(f"Number of Chunks: {len(chunks)}")
    print(f"Number of Similar Chunks: {not_diff}")
    print(f"Number of Unique Chunks Class One: {len(unique_chunks_set_class_one)}")
    print(f"Number of Unique Chunks Class Two: {len(unique_chunks_set_class_two)}")
    print(f"Number of unique chunks: {len(unique_chunks)}")
    return unique_chunks, unique_labels 


def remove_duplicates(train_features, train_labels, test_features, test_labels):
    unique_chunks_set_class_one = set()
    unique_chunks_set_class_two = set()
    not_diff = 0

    # for one class we just add to the set 
    def add_unique_chunks_set(chunks, labels):
        for chunk,label in zip(chunks, labels):
            test = "".join([str(x) for x in chunk])
            if label == 0:
                unique_chunks_set_class_one.add(test)
            else:
                unique_chunks_set_class_two.add(test)
    
    add_unique_chunks_set(train_features, train_labels)
    add_unique_chunks_set(test_features, test_labels)

    def unique_chunks_labels(chunks, labels):
        unique_chunks = []
        unique_labels = []
        for chunk, label in zip(chunks,labels):
            test = "".join([str(x) for x in chunk])
            if label == 0:
                if test not in unique_chunks_set_class_two:
                    unique_chunks.append(chunk)
                    unique_labels.append(label)
            else:
                if test not in unique_chunks_set_class_one:
                    unique_chunks.append(chunk)
                    unique_labels.append(label)
        return unique_chunks, unique_labels

    unique_train_features, unique_train_labels = unique_chunks_labels(train_features, train_labels)
    unique_test_features, unique_test_labels = unique_chunks_labels(test_features, test_labels)

    print(f"Number of Chunks: {len(train_features)+len(train_labels)}")
    print(f"Number of Unique Chunks Class One: {len(unique_chunks_set_class_one)}")
    print(f"Number of Unique Chunks Class Two: {len(unique_chunks_set_class_two)}")
    print(f"Number of unique chunks Training: {len(unique_train_labels)}")
    print(f"Number of unique chunks Testing: {len(unique_test_labels)}")
    return unique_train_features, unique_train_labels, unique_test_features, unique_test_labels 

def split_chunks(data, labels, chunk_size, skip=1):
    temp = []
    final_labels = []
    for sample_index, sample in enumerate(data):
        # get this at a chunk size 
        for index in range(0,len(sample)-chunk_size,skip):
            if index == 0:
                temp.append(sample[:chunk_size])
            else:
                temp.append(sample[index:index+chunk_size])
            final_labels.append(labels[sample_index])
    return temp, final_labels

def load_data(filepath, chunk_size, disease_first=True, skip=1):
    data = np.load(filepath, allow_pickle=True)
    print(data.shape)
    if disease_first:
        labels = [1]*len(sizes_disease)+[0]*len(sizes_control)
    else:
        labels = [0]*len(sizes_control)+[1]*len(sizes_disease)
    X,Y = split_chunks(data, labels, chunk_size, skip)
    # probably randomly shuffle this 
    combined = list(zip(X,Y))
    random.shuffle(combined)
    return zip(*combined)

def plot_list(alist, filename, points, display=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(alist)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.set_title(filename)
    
    # annotate specific point let use test with 
    # check if the index already occurs 
    visited = {}
    for point in points:
        index, name = point
        if index not in visited:
            ax.annotate(name, xy=(index, alist[index]), xycoords='data',
                        xytext=(-10, 60), textcoords='offset points',
                        bbox=dict(boxstyle="round", fc="0.6"),
                        arrowprops=dict(arrowstyle="->"))
            visited[index] = 20
        else:
            ax.annotate(name, xy=(index, alist[index]), xycoords='data',
                        xytext=(-10, 60+visited[index]), textcoords='offset points',
                        bbox=dict(boxstyle="round", fc="0.6"))         
            visited[index] += 20
    if display:
        plt.show()
    else:
        plt.savefig(filename)

def plot_lists(alist,alist2, filename, points, display=False, minor_freq=1, major_freq=20):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(alist)
    ax.plot(alist2, color="red")
    ax.set_xlabel("Epoches")
    ax.set_ylabel("Loss")
    ax.set_title(filename)
    
    # annotate specific point let use test with 
    # check if the index already occurs 
    visited = {}
    for point in points:
        index, name = point
        if index not in visited:
            ax.annotate(name, xy=(index, alist[index]), xycoords='data',
                        xytext=(-10, 60), textcoords='offset points',
                        bbox=dict(boxstyle="round", fc="0.6"),
                        arrowprops=dict(arrowstyle="->"))
            visited[index] = 20
        else:
            ax.annotate(name, xy=(index, alist[index]), xycoords='data',
                        xytext=(-10, 60+visited[index]), textcoords='offset points',
                        bbox=dict(boxstyle="round", fc="0.6"))         
            visited[index] += 20

    # Make a plot with major ticks that are multiples of 20 and minor ticks that
    # are multiples of 5.  Label major ticks with '%d' formatting but don't label
    # minor ticks.
    ax.set_xlim([0,len(alist)])
    ax.xaxis.set_major_locator(plt.MultipleLocator(major_freq))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_freq))
    major_labels = [0]+[i for i in range(0, len(alist)+1, minor_freq+1)]
    print(major_labels)
    ax.set_xticklabels(major_labels)

    if display:
        plt.show()
    else:
        plt.savefig(filename)


def plot_lists2(alist,alist2,alist3, filename, points, display=False, minor_freq=1, major_freq=20):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(alist)
    ax.plot(alist2, color="red")
    ax.plot(alist3, color="green")
    ax.set_xlabel("Epoches")
    ax.set_ylabel("Loss/Acccuracy")
    ax.set_title(filename)
    
    # annotate specific point let use test with 
    # check if the index already occurs 
    visited = {}
    for point in points:
        index, name = point
        if index not in visited:
            ax.annotate(name, xy=(index, alist[index]), xycoords='data',
                        xytext=(-10, 60), textcoords='offset points',
                        bbox=dict(boxstyle="round", fc="0.6"),
                        arrowprops=dict(arrowstyle="->"))
            visited[index] = 20
        else:
            ax.annotate(name, xy=(index, alist[index]), xycoords='data',
                        xytext=(-10, 60+visited[index]), textcoords='offset points',
                        bbox=dict(boxstyle="round", fc="0.6"))         
            visited[index] += 20

    # Make a plot with major ticks that are multiples of 20 and minor ticks that
    # are multiples of 5.  Label major ticks with '%d' formatting but don't label
    # minor ticks.
    ax.set_xlim([0,len(alist)])
    ax.xaxis.set_major_locator(plt.MultipleLocator(major_freq))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_freq))
    major_labels = [0]+[i for i in range(0, len(alist)+1, minor_freq+1)]
    print(major_labels)
    ax.set_xticklabels(major_labels)

    if display:
        plt.show()
    else:
        plt.savefig(filename)

# we will have accs as multiple number of accs 
def epoch_loss_plot(losses, accs, filename,points,accs2=None, display=False, 
                    major_freq=5, minor_freq=1, accs_name="Accuracy", 
                    accs2_name=None):
    # Plot two lines with different scales on the same plot
    fig, ax1 = plt.subplots(figsize=(15, 10))
    line_weight = 1
    alpha = .5
    #ax1 = fig.add_axes([0, 0, 1, 1])
    #ax2 = fig.add_axes()
    # This is the magic that joins the x-axis
    ax2 = ax1.twinx()
    lns1 = ax1.plot(losses, color='blue', lw=line_weight, alpha=alpha, label='Loss')
    lns2 = ax2.plot(accs, color='red', lw=line_weight, alpha=alpha, label=accs_name)
    leg = lns1 + lns2
    if accs2:
        lns3 = ax2.plot(accs2, color="green", lw=line_weight, alpha=alpha, label=accs2_name)
        leg += lns3
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc=0)
    plt.title(f'Epoch for Loss and Accuracy', fontsize=20, y=1.1)
    
    visited = {}
    for i,point in enumerate(points):
        if i == 0:
            index, name = point
            if index not in visited:
                ax2.annotate(name, xy=(index, accs[index]), xycoords='data',
                            xytext=(-10, 60), textcoords='offset points',
                            bbox=dict(boxstyle="round", fc="0.6"),
                            arrowprops=dict(arrowstyle="->"))
                visited[index] = 20
            else:
                ax2.annotate(name, xy=(index, accs[index]), xycoords='data',
                            xytext=(-10, 60+visited[index]), textcoords='offset points',
                            bbox=dict(boxstyle="round", fc="0.6"))         
                visited[index] += 20
        else:
            index, name = point
            if index not in visited:
                ax2.annotate(name, xy=(index, accs2[index]), xycoords='data',
                            xytext=(-10, 60), textcoords='offset points',
                            bbox=dict(boxstyle="round", fc="0.6"),
                            arrowprops=dict(arrowstyle="->"))
                visited[index] = 20
            else:
                ax2.annotate(name, xy=(index, accs2[index]), xycoords='data',
                            xytext=(-10, 60+visited[index]), textcoords='offset points',
                            bbox=dict(boxstyle="round", fc="0.6"))         
                visited[index] += 20

    # Make a plot with major ticks that are multiples of 20 and minor ticks that
    # are multiples of 5.  Label major ticks with '%d' formatting but don't label
    # minor ticks.
    ax1.set_xlim([0,len(losses)])
    ax1.xaxis.set_major_locator(plt.MultipleLocator(major_freq))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(minor_freq))
    
    # we know that accuracy is form 0 -> 1.0
    ax2.set_ylim([0,1])
    plt.tight_layout()
    if display:
        plt.show()
    else:
        plt.savefig(filename)
    fig.clear()
    plt.close(fig)

def test(filepath, chunk_size):
    data = np.load(filepath, allow_pickle=True)
    print(data.shape)
    labels = [0]*len(sizes_control)+[1]*len(sizes_disease)
    X,Y = split_chunks(data, labels, chunk_size)
    analyze_chunks(X, Y)

def convert_label_to_block(sample):
    bar = []
    color = []
    prev = sample[0]
    length = 1 
    for index in range(1,len(sample)):
        curr_sample = sample[index]
        if curr_sample == prev:
            length += 1 
        else:
            bar.append(length)
            color.append(str(prev))
            length = 1
        prev = curr_sample
    bar.append(length)
    color.append(str(prev))
    return bar, color

# could add a percentage of what is the blue and the red 
def final_plot(labels, samples, filename, add_label=10):
    data = [convert_label_to_block(x) for x in samples]
    values, orders = zip(*data)

    fig, ax = plt.subplots(figsize=(10,10)) 
    max_height = 0
    for index in range(len(values)):

        count_of_r_and_b = Counter(samples[index])
        count_r = count_of_r_and_b[0]
        count_b = count_of_r_and_b[1]

        for index2 in range(len(values[index])):
            height = sum(values[index][:index2])

            if height > max_height:
                max_height = height

            color = "r" 
            if orders[index][index2] == "1":
                color = "b"
            #width = values[index][index2]
            width = 0.35
            #print(f"index:{index}, height:{height}, width:{width}, color:{color}")
            name = color

            plt.barh(index, values[index][index2], width,left=height, color=color,label=orders[index][index2])

            if add_label and values[index][index2] > add_label:
                text_color = 'white'
                distance = height#+values[index][index2]/2
                ax.text(distance,index, str(values[index][index2]), ha='center', va='center',
                        color=text_color)
        ax.text(height+400,index, f"r:{count_r},b:{count_b}", ha='center', va='center',
                        color="black",  wrap=False)

    handles, bar_labels = ax.get_legend_handles_labels()
    i =1
    while i<len(bar_labels):
        if bar_labels[i] in bar_labels[:i]:
            del(bar_labels[i])
            del(handles[i])
        else:
            i +=1
    bottoms = np.arange(len(samples))
    ax.set_xlim([0,max_height+750])
    #plt.legend(handles, bar_labels, loc="best", bbox_to_anchor=(1.0, 1.00))
    # Put a legend to the right of the current axis
    plt.legend(handles, bar_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.yticks(bottoms, [f"Sample {t+1}: {labels[t]}" for t in bottoms])
    plt.savefig(filename)


# b: blue.
# g: green.
# r: red.
# c: cyan.
# m: magenta.
# y: yellow.
# k: black.
state_to_color = {
    "0" : "g",
    "1" : "c",
    "2" : "m",
    "3" : "y"
}

# could add a percentage of what is the blue and the red 
def plot_output_and_input(labels, samples, actual_input, filename, add_label=10):
    data = [convert_label_to_block(x) for x in samples]
    values, orders = zip(*data)

    data2 = [convert_label_to_block(x) for x in actual_input]
    values2, orders2 = zip(*data2)

    fig, ax = plt.subplots(figsize=(10,10)) 
    max_height = 0

    y_labels = []
    for index in range(len(values)):

        count_of_r_and_b = Counter(samples[index])
        count_r = count_of_r_and_b[0]
        count_b = count_of_r_and_b[1]

        bar_index = 2*index
        bar_index2 = 2*index + 1

        for index2 in range(len(values2[index])):
            height = sum(values2[index][:index2])
            # so I have a mapping for colors 
            state = orders2[index][index2]
            if state not in state_to_color:
                print(type(state))
                raise ValueError(f"{state} is not valid for mapping")
            color = state_to_color[state]
            width = 0.35
            name = color
            plt.barh(bar_index, values2[index][index2], width,left=height, color=color,label=state)


        for index2 in range(len(values[index])):
            height = sum(values[index][:index2])

            if height > max_height:
                max_height = height

            color = "r" 
            if orders[index][index2] == "1":
                color = "b"
            
            width = 0.35
            name = color
            plt.barh(bar_index2, values[index][index2], width,left=height, color=color,label=f"pred:{orders[index][index2]}")

            if add_label and values[index][index2] > add_label:
                text_color = 'white'
                distance = height#+values[index][index2]/2
                ax.text(distance,bar_index2, str(values[index][index2]), ha='center', va='center',
                        color=text_color)
        ax.text(height+400,bar_index2, f"r:{count_r},b:{count_b}", ha='center', va='center',
                        color="black",  wrap=False)
        
        y_labels.append(f"Sample X {index+1}: {labels[index]}")
        y_labels.append(f"Sample Y {index+1}: {labels[index]}")

    handles, bar_labels = ax.get_legend_handles_labels()
    i =1
    while i<len(bar_labels):
        if bar_labels[i] in bar_labels[:i]:
            del(bar_labels[i])
            del(handles[i])
        else:
            i +=1
    bottoms = np.arange(len(samples)*2)
    ax.set_xlim([0,max_height+750])
    # Put a legend to the right of the current axis
    plt.legend(handles, bar_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.yticks(bottoms, y_labels)
    plt.savefig(filename)
    fig.clear()
    plt.close(fig)

def check_file(filepath):
    return path.exists(filepath)

def calc_balance_acc(sample_preds, labels):
    y_true = []
    y_pred = []
    for index,sample in enumerate(sample_preds):
        y_true.extend([labels[index]]*len(sample))
        y_pred.extend(sample)
    assert len(y_true) == len(y_pred)
    return balanced_accuracy_score(y_true, y_pred)


band_to_lengths = {
'delta': [623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 
           623, 623, 623, 623, 623, 623, 623, 617, 623, 623, 623, 623, 623, 623, 
           617, 623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 
           623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 
           623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 623, 623], 
'theta': [712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 
          712, 712, 712, 712, 712, 712, 712, 706, 712, 712, 712, 712, 712, 712, 
          706, 712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 
          712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 
          712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 712, 712], 
'alpha': [711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 
          711, 711, 711, 711, 711, 711, 711, 705, 711, 711, 711, 711, 711, 711, 
          705, 711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 
          711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 
          711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 711, 711], 
'beta': [3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 
         3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 3176, 3203, 3203, 
         3203, 3203, 3203, 3203, 3176, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 
         3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 
         3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203, 
         3203, 3203, 3203, 3203, 3203, 3203, 3203, 3203], 
'gamma': [12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453, 
          12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453, 
          12453, 12349, 12453, 12453, 12453, 12453, 12453, 12453, 12349, 12453, 
          12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453, 
          12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453, 
          12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453, 
          12453, 12453, 12453, 12453, 12453, 12453, 12453, 12453]}

def split_samples(filepath, sizes):
    microstates = np.load(filepath)
    return [list(islice(microstates, elem)) for elem in sizes] 

def write_to_file(filepath, num_states=100, disease_first=True):
    data = np.load(filepath, allow_pickle=True)
    
    if disease_first:
        with open("patients_w_pd.txt", "w") as f:
            for index in range(len(sizes_disease)):
                f.write("".join([str(data[index][x]) for x in range(num_states)])+"\n")
        with open("controls.txt", "w") as f:
            for index in range(len(sizes_disease), len(sizes_disease)+len(sizes_control)):
                f.write("".join([str(data[index][x]) for x in range(num_states)])+"\n")
    else:
        with open("controls.txt", "w") as f:
            for index in range(len(sizes_control)):
                f.write("".join([str(data[index][x]) for x in range(num_states)])+"\n")
        with open("patients_w_pd.txt", "w") as f:
            for index in range(len(sizes_control), len(sizes_disease)+len(sizes_control)):
                f.write("".join([str(data[index][x]) for x in range(num_states)])+"\n")
    print("Done")

if __name__ == "__main__":
    # test("data/sample_peaks_4.npy", 10)
    # test("data/sample_peaks_4.npy", 15)
    # test("data/sample_peaks_4.npy", 20)
    # test("data/sample_peaks_4.npy", 25)
    # test("data/sample_peaks_4.npy", 26)
    # test("data/sample_peaks_4.npy", 27)
    # test("data/sample_peaks_4.npy", 28) # we will stay at 28
    # test("data/sample_peaks_4.npy", 29)
    # test("data/sample_peaks_4.npy", 30)
    # test("data/peaks_with_dist_4.npy", 15)

    # losses = [100*x for x in range(9,0,-1)]
    # accs = [x*0.1 for x in range(0,10)]
    # accs2 = [(x+1)*0.1 for x in range(0,10)]
    # points = [(5,"hello"), (6,"mr.kenobi")]
    # epoch_loss_plot(losses, accs, "test", points, accs2=accs2, display=True,accs2_name="Accuracy2")
    #final_plot([0], [[0,1,0,1,0,0,1,0,1]], "dumb", add_label=10)
    for band in band_to_lengths:
        output = split_samples(f"data/microstates2_{band}_4.npy", band_to_lengths[band])
        np.save(f"samples_{band}_4", output)

    #write_to_file("data/sample_peaks_4.npy", num_states=100, disease_first=True)