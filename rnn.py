#
# Author: Normand Overney
#

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import time
import math
import random

# to plot the loss function
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# use sklearn to split the data 
from sklearn.model_selection import train_test_split
# to deal with imbalance by down or up sampling
from sklearn.utils import resample
# for the sack of counting the predictions 
from collections import Counter

# this is to use the GPU once I have it running at home 
use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

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


def split_microstates(sequence, sizes):
    seqs = []
    index = 0
    for size in sizes:
        seqs.append(sequence[index:index+size])
        index += size
    return seqs

def up_sample(train_features, train_labels, random_state, print_all):
    # since I am not using one hot encoding
    class_one = train_features[train_labels[:]==0]
    class_two = train_features[train_labels[:]==1]
    
    if print_all:
        print(f"Number of class one: {len(class_one)}")
        print(f"Number of class two: {len(class_two)}")

    # we need to upsample the class 0 
    class_one_upsamples = resample(class_one,
                                   replace=True, # sample with replacement
                                   n_samples=len(class_two), # match number in majority class
                                   random_state=random_state) # reproducible results
    
    if print_all:
        print(f"Number of class one upsampled: {len(class_one_upsamples)}")
    # next we just concatinate back again
    train_features = np.concatenate([class_one_upsamples, class_two], axis=0)
    train_labels = np.concatenate([[0]*len(class_one_upsamples), [1]*len(class_two)], axis=0)
    return train_features, train_labels

# Just for demonstration, turn a letter into a <1 x n_states> Tensor
def stateToTensor(state, n_states=4):
    # change to cuda
    tensor = torch.zeros(1, n_states)
    if use_cuda:
        tensor = torch.cuda.FloatTensor(1,n_states).fill_(0)
    tensor[0][state] = 1
    return tensor

# Turn a chunk of states into a <chunk_length x 1 x n_states>,
# or an array of one-hot letter vectors
def chunkToTensor(chunk, n_states=4):
    # change to cuda
    tensor = torch.zeros(len(chunk), 1, n_states)
    if use_cuda:
        tensor = torch.cuda.FloatTensor(len(chunk), 1, n_states).fill_(0)
    for li, state in enumerate(chunk):
        tensor[li][0][state] = 1
    return tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        init = Variable(torch.zeros(1, self.hidden_size))
        if use_cuda:
            init =  Variable(torch.zeros(1, self.hidden_size)).cuda()
        return init

def train(rnn, category_tensor, line_tensor, criterion, optimizer):
    rnn.zero_grad()
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # this has been obsoleted for the sake of 
    optimizer.step()
    #print(loss.data)
    #we had to modify the return of loss.data[0]
    return output, loss.data

# so maybe I need the random training sample 
def randomChoice(l):
    return random.randint(0, len(l) - 1)

# he get random from 
def randomTrainingExample(sequences, chunk_size):
    sample_index = randomChoice(sequences)
    chunk_index = randomChoice(sequences[sample_index])
    category = 0
    if sample_index >= len(sizes_control):
        category = 1
    # we randomly choose a chunk of size ten 
    chunk = sequences[sample_index][chunk_index:chunk_index+chunk_size]
    category_tensor = torch.tensor([category], dtype=torch.long).cuda()
    chunk_tensor = chunkToTensor(chunk)
    return category, chunk, category_tensor, chunk_tensor

def getTrainingExample(train_features, train_labels, sample_index, chunk_index, chunk_size):
    category = train_labels[sample_index]
    # we randomly choose a chunk of size ten 
    chunk = train_features[sample_index][chunk_index:chunk_index+chunk_size]
    assert len(chunk) == chunk_size, str(chunk)
    category_tensor = torch.tensor([category], dtype=torch.long)
    if use_cuda:
        category_tensor = torch.tensor([category], dtype=torch.long).cuda()
    chunk_tensor = chunkToTensor(chunk)
    return category, chunk, category_tensor, chunk_tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i, category_i

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# so I will get this working in pyroth since I have way too much free time
# load the sequence 
def run(train_features, train_labels, n_hidden, chunk_size, to_plot):
    # these should be hardcoded 
    n_categories = 2
    n_states = 4 # number of states

    rnn = RNN(n_states, n_hidden, n_categories)
    # to convert to GPU or CPU
    rnn.to(device)
    # inputs now this is where we decide what we need to classify maybe
    # have seq of time
    criterion = nn.NLLLoss()

    # If too low, it might not learn
    # If you set this too high, it might explode.
    learning_rate = 0.005
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    # how many epoches here to go through all samples 
    # the length of the training samples is 
    n_iters = sum([len(x) for x in train_features]) - len(train_features)*chunk_size
    print(f"Number of Iterations:{n_iters}")
    #n_iters = 100
    # we are a farctor ten off of getting everything processed if we do sliding by 100000
    sample_index = 0
    chunk_index = 0

    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    # change from control index and patient index 
    # not sure if I event need this 
    flip_sample = True
    sample_index_control = 0
    sample_index_patient = len(train_features)//2
    chunk_index_control = 0
    chunk_index_patient = 0

    for iter in range(1, n_iters + 1):
        # so if it is control
        if flip_sample:
            flip_sample = False

            if len(train_features[sample_index_control][chunk_index_control:chunk_index_control+chunk_size]) < chunk_size:
                sample_index_control += 1 
                chunk_index_control = 0
                if sample_index_control == len(train_features)//2:
                    sample_index_control = 0 

            sample_index = sample_index_control
            chunk_index = chunk_index_control

            chunk_index_control += 1
        else:
            flip_sample = True 

            if len(train_features[sample_index_patient][chunk_index_patient:chunk_index_patient+chunk_size]) < chunk_size:
                print("I AM HERE")
                sample_index_patient += 1 
                chunk_index_patient = 0
                if sample_index_patient == len(train_features):
                    sample_index_patient = len(train_features)//2

            sample_index = sample_index_patient
            chunk_index = chunk_index_patient

            chunk_index_patient += 1
        """
        print(sample_index)
        print(chunk_index)
        print(sample_index_control)
        print(sample_index_patient)
        print(chunk_index_control)
        print(chunk_index_patient)
        """
        
        # we just do the things neccesary
        category, line, category_tensor, line_tensor = getTrainingExample(train_features, train_labels, sample_index, chunk_index, chunk_size)
        output, loss = train(rnn, category_tensor, line_tensor, criterion, optimizer)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
            print(f'Sample Index:{sample_index}')
            print(f'Chunk Index:{chunk_index}')
            print(f"Current Loss:{current_loss}")
        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    

    if to_plot:
        plt.figure()
        plt.plot(all_losses)
        plt.show()

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])


    return rnn

# Just return an output given a line
def evaluate(rnn, chunk_tensor):
    hidden = rnn.init_hidden()
    
    for i in range(chunk_tensor.size()[0]):
        output, hidden = rnn(chunk_tensor[i], hidden)
    
    return output

# the number of predictions has to be for the number of available classes
def predict(rnn, input_chunk, n_predictions=2, to_display=False):
    output = evaluate(rnn, Variable(chunkToTensor(input_chunk)))
    if use_cuda:
        output = evaluate(rnn, Variable(chunkToTensor(input_chunk)).cuda())

    #print('\n> %s' % input_chunk)

    all_categories = [0, 1]

    # Get top N categories
    topv, topi = output.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i].item()
        category_index = topi[0][i].item()
        if to_display:
            print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, category_index])

    return predictions



def calculate_accuracy(rnn, test_features, test_labels, chunk_size):
    total_accuracy = 0 
    control_accuracy = 0
    control_count = 0
    patient_accuracy = 0
    patient_count = 0
    # I want to see how to go through and average 
    for sample, sample_label in zip(test_features, test_labels):
        predictions = []
        for i in range(len(sample) - chunk_size):
            predictions.append(predict(rnn, sample[i:i+chunk_size])[0][1])
        print(Counter(predictions), sample_label)
        results = Counter(predictions)
        if max(results.values()) == results[sample_label]:
            total_accuracy += 1 
            if sample_label == 1:
                patient_accuracy += 1
            elif sample_label == 0:
                control_accuracy += 1        
        if sample_label == 1:
            patient_count += 1
        else:
            control_count += 1
    print(f"Total Accuracy: {100*total_accuracy/len(test_features)}")
    print(f"Patient Accuracy: {100*patient_accuracy/patient_count}")
    print(f"Control Accuracy: {100*control_accuracy/control_count}")
    
    # we can do the calculations later 

# need to split the data to training and testing data 
def main(args):
    filepath = args['filepath']
    sequences = [] # this should be the same for both 
    if args["all_microstates"]:
        microstates = np.load(args['filepath'])
        sequences = split_microstates(microstates, sizes_control+sizes_disease)
    else:
        sequences = np.load(filepath, allow_pickle=True)
        print(len(sequences))
        print(len(sequences[0]))

    #for i in range(10):
    #    category, line, category_tensor, line_tensor = randomTrainingExample(sequences, args['chunk_size'])
    #    print('category =', category, '/ line =', line)
    test_size = 0.25
    random_state = 42
    X = np.array(sequences)
    Y = np.array([0]*len(sizes_control)+[1]*len(sizes_disease))
    # we need to split the data 
    train_features, test_features, train_labels, test_labels = train_test_split(X,Y, 
                                                               test_size = test_size,
                                                               random_state = random_state)
    print("\nOriginal Sampled Data:")
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    train_features, train_labels = up_sample(train_features, train_labels, random_state, True)

    print("\nNew Upsampled Data:")
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)
    
    train = False
    if train:
        rnn = run(train_features, train_labels, args['hidden'],  args['chunk_size'], args['to_plot'])

        # we want to save the model since it took forever to train it 
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in rnn.state_dict():
            print(param_tensor, "\t", rnn.state_dict()[param_tensor].size())

        # to save the model 
        torch.save(rnn, "simple_rnn.pt")

    rnn = torch.load("simple_rnn.pt")
    # to convert to GPU or CPU
    rnn.to(device)
    calculate_accuracy(rnn, test_features, test_labels, args['chunk_size'])


# Notes:
# - errors seems to mostly in predictiny 0 since we have imbalanced data so I should 
# - just grab 20 from each subgroups 
# - we then have 4 samples for testing controls that is a bit too less but still 16% 
# - takes an hour for 100000 iterations 
# - so with 1209420 length of sequences at 100 chunk we should have scanned through a 
# - a tenth of all possible sequences 
#
# TODO:
# the chunks beyond ten don't really make sense 
# work on a bash script to make it run multiple version
# need to encode the space in between as a seperate class -> talk with Volker dont need to do that 
# 20 min to do one epoch and that is including the upsampling 
# 
if __name__ == "__main__":
    args = {'filepath' : "data/sample_peaks_4.npy",
            'to_plot':True, 'chunk_size': 25,
            'all_microstates':False,
            'hidden':128}
    main(args)