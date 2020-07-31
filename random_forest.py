#
# Author: Normand Overney 
# 
# Notes for Random Forest:
# - building blocks for random forest are decision trees 
#   - split the data such resulting groups are as different from each other as possible 
#   - the members have to be as similiar as possible 
# - large number of decision trees that operate as an emsemble 
# - each individual tree in a random forest spits out a class prediction then we take the class with
#   the most votes 
# - large number of uncorrelated models opperating together outperforms any individual constituent models
# - trees usally have the same individual errors 
# - Two prereq for good random forests 
#   1. actually have some feaures which make it such that a model can perfom better than random quessing
#   2. predictions (and errors) made by the individual trees need to have low correlations with each other 
# - number of correct predictions increase with the number of uncorrelated trees
# - Bagging (bootstrap aggregation) decision trees sensitive to data that they are trained on 
#   - each individual tree to randomly sample from the dataset with replacement => different trees 
#   - with bagging there is no subsetting the training data into smaller chunks 
#   - train each tree on a different chunk
#   - e.g. training data [1,2,3,4,5,6] we give a tree [1,2,2,3,6,6] (they are voth length six)
# - Feature Randomness
#   Normal Descision Tree:
#   - when we decide to split, we look at every single feature 
#   - we choose the feature which produces the most seperation between left and right nodes
#   Random Forest
#   - can only pick from a random subset of features 
#   - forces trees to be different from each other 
#   - allows for lower correlation and more diversification among trees
# - random forest has trees trained from different sets of data and use diffrent features to make decisions 
# - when we put garbage in we get garbage out but if we do better than random 
#
# How do we deal with balance?
# - most machine learning algorithms work better when the number of samples in each class are about equal 
# - probably create a confusion matrix showing the correct predictions and the types of incorrect predictions
# - precision => want to know the number of false positives
# - recall => the number of true positives dived by the number of positive values 
#   measure of completeness
# - F1 score: weighted average of precision and recall
# - Resample Techniques - Oversampl minority class
#   - adding more copies of the minority class
#   - good for when there is simple no data available 
#   - always split your daa into test and train sets before ovrsampling techniques then we wll just overfitting
#   - we can also downsample the larger one but this results in just underfitting and poor generalization to test set
#   - generate synthetic samples -> that seems like a pretty good idea 
# 1. Change the performance metric
# 2. Change the algorithm
# 3. Oversample minority class
# 4. Undersample majority class
# 5. Generate synthetic samples
import numpy as np
from sortedcontainers import SortedDict
import time
import pandas as pd

# use sklearn to split the data 
from sklearn.model_selection import train_test_split

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import export_graphviz
import pydot

# to deal with imbalance by down or up sampling
from sklearn.utils import resample

# in order to generate synthetic samples
from imblearn.over_sampling import SMOTE

# such that we can run a bunch of things in parallel 
import argparse

import os

# these are too be hard coded 
sizes_control = [177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 
                  177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900]

# for disease we have a few files which are 176400 long and there are only two of them 
# so missing only 1500 time steps which is just 1.5 seconds 
sizes_disease = [177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 
                 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 
                 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 177900, 
                 177900, 176400, 177900, 176400, 177900]

# we have to divide everything by ten 
SIZES_CONTROL = [int(x/10) for x in sizes_control]
SIZES_DISEASE = [int(x/10) for x in sizes_disease]

def read_file(filepath):
    return np.load(filepath)


def create_2nd_order_transM(sequence):
    # I will store it as a dictionary 
    # so I have <class1><class2> -> <class3>
    words = SortedDict()
    # we need a two class scanner 
    for index in range(len(sequence)-2):
        key = "{0}+{1}".format(sequence[index], sequence[index+1])
        next_letter = sequence[index+2]
        # we need another dictionary in the dictionary 
        if key in words:
            if next_letter in words[key]:
                words[key][next_letter] += 1
            else:
                words[key][next_letter] = 1
                
        else:
            words[key] = SortedDict({next_letter:1})
    # we divide by the number of possible transitions we
    for key in words:
        for letter in words[key]:
            words[key][letter] /= (len(sequence)-2.0)
    return words

def create_1st_order_transM(sequence):
    words = SortedDict()
    for index in range(len(sequence) -1):
        key = sequence[index]
        next_letter = sequence[index+1]
        if key not in words:
            words[key] = SortedDict({next_letter:1})
        else:
            if next_letter not in words[key]:
                words[key][next_letter] = 1
            else:
                words[key][next_letter] += 1
    for key in words:
        for letter in words[key]:
            words[key][letter] /= (len(sequence)-1.0)
    return words

def transM_to_table(transM):
    table = []
    for key in transM:
        table.extend(transM[key].values())
    return table


def get_features(transM):
    features = []
    for key in transM:
        for innerKey in transM[key]:
            features.append(f"{key}->{innerKey}")
    return features

def generate_x_and_y(microstates):
    all_samples = SIZES_CONTROL + SIZES_DISEASE
    length = 0

    first_order_X = []
    second_order_X = []
    Y = [0]*len(SIZES_CONTROL) + [1]*len(SIZES_DISEASE)

    for sample_length in all_samples:
        temp = microstates[length:length+sample_length]
        first_order = create_1st_order_transM(temp)
        second_order = create_2nd_order_transM(temp)
        first_order_X.append(transM_to_table(first_order))
        second_order_X.append(transM_to_table(second_order))
        length += sample_length

    first_order_X = np.array(first_order_X)
    second_order_X = np.array(second_order_X)
    Y = np.array(Y)

    # get the feature names 
    feature_X1 = get_features(first_order)
    feature_X2 = get_features(second_order)

    return {"first_order_X"  : first_order_X, 
            "feature_X1"     : feature_X1,
            "second_order_X" : second_order_X, 
            "feature_X2"     : feature_X2,
            "Y"              : Y}

def down_sample(train_features, train_labels, random_state, print_all):
    X = np.concatenate([train_features, train_labels], axis=1)
    # separate minority and majority classes
    class_one = X[X[:,-2]==1]
    class_two = X[X[:,-1]==1]
    
    if print_all:
        print(f"Number of class one: {len(class_one)}")
        print(f"Number of class two: {len(class_two)}")

    assert len(class_one) > 0, "number of class one is zero"
    assert len(class_two) > 0, "number of class two is zero"
    # we need to upsample the class 0 
    class_two_downsamples = resample(class_two,
                                   replace=True, # sample with replacement
                                   n_samples=len(class_one), # match number in majority class
                                   random_state=random_state) # reproducible results
    
    if print_all:
        print(f"Number of class one downsampled: {len(class_two_downsamples)}")
    # next we just concatinate back again
    train_features = np.concatenate([class_two_downsamples[:,:-2], class_one[:,:-2]], axis=0)
    train_labels = np.concatenate([class_two_downsamples[:,-2:], class_one[:,-2:]], axis=0)
    return train_features, train_labels

def up_sample(train_features, train_labels, random_state, print_all):
    X = np.concatenate([train_features, train_labels], axis=1)
    # separate minority and majority classes
    class_one = X[X[:,-2]==1]
    class_two = X[X[:,-1]==1]
    
    if print_all:
        print(f"Number of class one: {len(class_one)}")
        print(f"Number of class two: {len(class_two)}")

    assert len(class_one) > 0, "number of class one is zero"
    assert len(class_two) > 0, "number of class two is zero"
    # we need to upsample the class 0 
    class_one_upsamples = resample(class_one,
                                   replace=True, # sample with replacement
                                   n_samples=len(class_two), # match number in majority class
                                   random_state=random_state) # reproducible results
    
    if print_all:
        print(f"Number of class one upsampled: {len(class_one_upsamples)}")
    # next we just concatinate back again
    train_features = np.concatenate([class_one_upsamples[:,:-2], class_two[:,:-2]], axis=0)
    train_labels = np.concatenate([class_one_upsamples[:,-2:], class_two[:,-2:]], axis=0)
    return train_features, train_labels

def generate_synthetic_samples(X, Y, random_state):
    sm = SMOTE(random_state=random_state)#, ratio=1.0)
    return sm.fit_sample(X, Y)

# might have to have the test size modified 
def algorthm(X,Y, feature_list, tree_name, sample="syn", test_size=0.25, 
             random_state=42, print_all=False):
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(X,Y, 
    test_size = test_size, random_state = random_state)

    if sample == "up":
        train_features, train_labels = up_sample(train_features, train_labels, 
                                                 random_state, print_all)
    elif sample == "down":
        train_features, train_labels = down_sample(train_features, train_labels,
                                                   random_state, print_all)
    elif sample == "syn":
        pass
    else:
        print("Valid Sampling Method was not chosen may have imbalance")

    if print_all:
        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', test_labels.shape)

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = random_state)
    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    
    # I do not consider a fifty percent correct since no way to accurate 
    # determine the class
    right = 0
    accuracy = predictions * test_labels
    predictions_vals = []
    for val in accuracy.flatten():
        if val != 0 and val > 0.5:
            right += 1
            predictions_vals.append(val)
        elif val != 0:
            predictions_vals.append(val)

    # MSE
    se = np.square(test_labels - predictions)
    mse = 0
    for row in range(predictions.shape[1]):
        mse += np.sum(se[:,row])/len(test_features)
    print(f"MSE: {mse}")

    # Accuracy 
    print(f"Total Accuaracy: {100*right/len(test_features)}")
    # print(f"{right}/{len(test_features)}")
    # print(predictions_vals)

    # we need to print the accuracy for each class since it is imbalanced
    accuracy_class_one = 0
    number_class_one = 0
    accuracy_class_two = 0
    number_class_two = 0


    for row_index in range(len(test_labels)):
        pred = predictions[row_index,]
        label = test_labels[row_index,]
        # class one 
        # it is one hot encoding so whatever
        if label[0] == 1:
            if pred[0] > 0.5:
                accuracy_class_one += 1
            number_class_one += 1
        else:
            if pred[1] > 0.5:
                 accuracy_class_two += 1
            number_class_two += 1
    
    assert number_class_one > 0, "number of class one is zero"
    assert number_class_two > 0, "number of class two is zero"

    print(f"Accuracy of Controls: {100*accuracy_class_one/number_class_one}")
    print(f"Accuracy of Patients with PD: {100*accuracy_class_two/number_class_two}")
    # Pull out one tree from the forest

    if print_all:
        tree = rf.estimators_[5]
        # Export the image to a dot file
        export_graphviz(tree, out_file = tree_name+'.dot', feature_names = feature_list, rounded = True, precision = 1)
        # Use dot file to create a graph
        (graph, ) = pydot.graph_from_dot_file(tree_name+'.dot')
        # Write graph to a png file
        graph.write_png(tree_name+'.png')

def main(args):
    print(f"\nRandom State: {args.random_state}, Sampling: {args.sampling}, Test Size: {args.test_size}")


    part_name = f"_{args.random_state}_{args.sampling}_{args.test_size}"
    first_tree_name =  os.path.join(args.outpath, "first_order_tree" + part_name) 
    second_tree_name =  os.path.join(args.outpath, "second_order_tree" + part_name) 

    start = time.time()
    random_state = args.random_state
    microstates = read_file(args.filepath)
    sampling_type = args.sampling
    print_all = args.print

    # TODO: should I play with test size 
    test_size = args.test_size

    if print_all:
        print(f"Length of microstates:{len(microstates)}")
        print(f"Total lengths of sampels:{sum(SIZES_CONTROL)+sum(SIZES_DISEASE)}")

    info = generate_x_and_y(microstates)

    X1          = info["first_order_X"]
    features_X1 = info["feature_X1"]
    X2          = info["second_order_X"]
    features_X2 = info["feature_X2"]
    Y           = info["Y"]

    if print_all:
        print("\n", f"first-order X: {X1.shape}", 
                    f"second-order X: {X2.shape}", 
                    f"Y: {Y.shape}")
    
    if args.sampling == "syn":
        X1,Y1 = generate_synthetic_samples(X1, Y, random_state)
        X2,Y2 = generate_synthetic_samples(X2, Y, random_state)
        if print_all:
            print("\n", f"first-order X: {X1.shape}", 
                f"second-order X: {X2.shape}", 
                f"Y: {Y.shape}")
    # One-hot encode the data using pandas get_dummies
    if args.sampling == "syn":
        features1 = np.array(pd.get_dummies(Y1))
        features2 = np.array(pd.get_dummies(Y2))
    else:
        features1 = np.array(pd.get_dummies(Y))
        features2 = np.array(pd.get_dummies(Y))

    # class one is the controls 
    # class two is the patients with pd 
    # now that we have the data we get the random forest 
    print("First Order:")
    algorthm(X1,features1,features_X1, first_tree_name, test_size=test_size,
             random_state=random_state, sample=sampling_type, print_all=print_all)
    print()
    print("Second Order:")
    algorthm(X2,features2,features_X2, second_tree_name, test_size=test_size,
             random_state=random_state, sample=sampling_type, print_all=print_all)
    if args.print:
        print(f"Total Time: {time.time()-start:.4f}")


# 
# Best classification = synthetic plus up sampling 
# First Order:
# Accuaracy: 68.18181818181819
# Accuracy of Controls: 57.142857142857146
# Accuracy of Patients with PD: 87.5
# Second Order:
# Accuaracy: 86.36363636363636
# Accuracy of Controls: 78.57142857142857
# Accuracy of Patients with PD: 100.0
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Forest Baseline')
    # maybe require them to be valid paths
    parser.add_argument('--filepath', default="data/all_microstates_tenth.npy", type=str)
    parser.add_argument('--outpath', default="data", type=str)
    parser.add_argument('--random_state', type=int, required=True)
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--sampling', type=str, help="syn: synthetic, up: upsampling, and down: downsampling")
    parser.add_argument('--print', default=False, action='store_true')
    args = parser.parse_args()
    main(args)