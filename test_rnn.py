#
# Author: Normand Overney
#

import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import numpy as np
import time
# use sklearn to split the data 
from sklearn.model_selection import train_test_split
import random

from utils import generate_synthetic_samples
from utils import generate_synthetic_samples_alt
from utils import upsample

from utils import sizes_control, sizes_disease
from utils import epoch_loss_plot
from utils import plot_output_and_input
from utils import check_file
from utils import print_class_number
from utils import calc_balance_acc

# get counter 
from collections import Counter
import ntpath
# make sure the cuda works 
from torch.autograd import Variable

# we do want to swtich from gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Pytorch device: {device}")

# set the random seed 
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# add such that it have not just a sliding window 
def split_chunks(data, labels, chunk_size, skip=0):
    temp = []
    final_labels = []
    for sample_index, sample in enumerate(data):
        # get this at a chunk size 
        for index in range(len(sample)-chunk_size-skip):
            if index == 0:
                temp.append(sample[:chunk_size])
            else:
                temp.append(sample[index+skip:index+skip+chunk_size])
            final_labels.append(labels[sample_index])
    return temp, final_labels


def fill_last_chunk(block, labels):
    print(len(block))
    lengths = []
    counts = Counter(labels)
    for sample,label in zip(block, labels):
        lengths.append(len(sample))
    # just makes the block to fifty 
    print(counts)

def block_data(train_features, train_labels, chunk_size, blocks, random_state):
    temp, final_labels = split_chunks(train_features, train_labels, chunk_size, skip=chunk_size//2)
    start = time.time()
    temp, final_labels = upsample(np.array(temp), np.array(final_labels), random_state)
    print(f"Clustering time: {time.time()-start}")
    # we need to randomly shuffle the lists 
    combined = list(zip(temp, final_labels))
    #random.shuffle(combined)

    temp,final_labels = zip(*combined)
    # probably should use some up-sampling or synthetic sampling 
    print(f"Number of Samples: {len(final_labels)}")

    # I need to bundle it into blocks 
    temp = [temp[i:i+blocks] for i in range(0,len(temp), blocks)]
    final_labels = [final_labels[i:i+blocks] for i in range(0,len(final_labels), blocks)]
    # we cannot use 
    # the last one since it is not perfectly blocked 
    # zero fill the last one aka add a number not existent 
    # hard code it to be -1 since most things will be above zero 
    fill_last_chunk(temp[-1], final_labels[-1]) # this gets no use since a rerun of data # does not replicate it
    temp = temp[:-1]
    final_labels = final_labels[:-1]

    # I need to shuffle the data here 
    X = np.array(temp)
    Y = np.array(final_labels)
    X = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))
    Y = Y.reshape((Y.shape[0]*Y.shape[1],))
    perm = np.random.permutation(len(X))
    #np.take(X, np.random.permutation(X.shape[axis]), axis=axis, out=X)
    #np.take(Y, np.random.permutation(Y.shape[axis]), axis=axis, out=Y)
    print(X.shape)
    print(Y.shape)
    X = X[perm]
    Y = Y[perm]

    X = X.reshape((X.shape[0]//blocks,blocks,chunk_size))
    Y = Y.reshape((Y.shape[0]//blocks,blocks))
    print(X.shape)
    print(Y.shape)
    #quit()
    # temp = torch.Tensor(temp)
    # final_labels = torch.LongTensor(final_labels)
    # print(temp.size())
    # print(final_labels.size())
    return X, Y

def create_data(filepath,chunk_size=10, blocks=100, random_state=42, disease_first=True):
    data = np.load(filepath, allow_pickle=True)
    print(data.shape)

    labels = [0]*len(sizes_control)+[1]*len(sizes_disease)
    if disease_first:
        labels = [1]*len(sizes_disease)+[0]*len(sizes_control)
    test_size = 0.20
    train_features, test_features, train_labels, test_labels = train_test_split(data,labels, 
                                                               test_size = test_size,
                                                               random_state = random_state)
    
    train_features, train_labels = block_data(train_features, train_labels, 
                                              chunk_size, blocks, random_state)

    return train_features, test_features, train_labels, test_labels

class RNNModelMultiLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_size=100, dropout=0.10):
        # we are just taking the first RNN 
        super(RNNModelMultiLayer, self).__init__()

        self.rnns = []
        self.vars = []
        self.dropouts = []
        input_dim = input_dim
        p = dropout
        for index, hidden_dim in enumerate(hidden_dims):
            self.rnns.append(nn.RNN(input_dim, hidden_dim, 1, batch_first=True, 
                                    nonlinearity='relu').cuda())
            self.vars.append(Variable(torch.zeros(1, batch_size, hidden_dim).type(torch.cuda.FloatTensor), 
                             requires_grad=True))
            self.dropouts.append(nn.Dropout(p=p))
            print(p)
            p /= (2*(index+1))
            input_dim = hidden_dim
        
        self.fc = nn.Linear(input_dim, output_dim).cuda()
        hidden_dims_name = "_".join([str(x) for x in hidden_dims])
        # add a name for print reasons
        self.name = f"RNNModelMultiLayer_{hidden_dims_name}_{int(dropout*100)}"

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        out = x.view(x.size(0),1,x.size(1))
        for index, (var, rnn) in enumerate(zip(self.vars, self.rnns)):
            out, hn = rnn(out, var.detach())
            out = self.dropouts[index](out)
        # Index hidden state of last time step
        # just want last time step hidden states! 
        out = self.fc(out[:, -1, :])
        return out

class LSTMModelMultiLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_size=100, dropout=0.10):
        # we are just taking the first RNN 
        super(LSTMModelMultiLayer, self).__init__()

        self.lstms = []
        self.vars = []
        self.biases = []
        self.dropouts = []
        input_dim = input_dim
        p = dropout
        for index, hidden_dim in enumerate(hidden_dims):
            self.lstms.append(nn.LSTM(input_dim, hidden_dim, 1, batch_first=True).cuda())
            self.vars.append(Variable(torch.zeros(1, batch_size, hidden_dim).type(torch.cuda.FloatTensor), 
                             requires_grad=True))
            self.biases.append(Variable(torch.zeros(1, batch_size, hidden_dim).type(torch.cuda.FloatTensor), 
                             requires_grad=True))
            self.dropouts.append(nn.Dropout(p=p))
            print(p)
            p /= (2*(index+1))
            input_dim = hidden_dim
        
        self.fc = nn.Linear(input_dim, output_dim).cuda()
        hidden_dims_name = "_".join([str(x) for x in hidden_dims])
        # add a name for print reasons
        self.name = f"LSTMModelMultiLayer_{hidden_dims_name}_{int(dropout*100)}"

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        out = x.view(x.size(0),1,x.size(1))
        for index, (var, lstm) in enumerate(zip(self.vars, self.lstms)):
            c = self.biases[index]
            out, (hn, cn) = lstm(out, (var, c))
            out = self.dropouts[index](out)
        # Index hidden state of last time step
        # just want last time step hidden states! 
        out = self.fc(out[:, -1, :])
        return out



class RNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layer_dim=1, 
                 batch_size=100, dropout=0.10):
        # we are just taking the first RNN 
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim
        self.batch_dim = batch_size

        # Building your RNN
        #batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, input_dim)
        # batch_dim = number of samples per batch
        # try different dropout

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True,
                          dropout=dropout, nonlinearity='relu').cuda()

        # add a bidirectional GRU
        #self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, 
        #                  batch_first=True, bidirectional=True, dropout=0.25).cuda() 
        # wait shouldn't I try to have more layers before I hit the read out layer
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim).cuda()

        # add a name for print reasons
        self.name = f"RNNModel_{hidden_dim}_{layer_dim}_{int(dropout*100)}"

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        x = x.view(x.size(0),1,x.size(1))
        h0 = Variable(torch.zeros(self.layer_dim, self.batch_dim, self.hidden_dim).type(torch.cuda.FloatTensor), requires_grad=True)
        #h0 = torch.zeros(self.layer_dim, self.batch_dim, self.hidden_dim).requires_grad_()

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        #print(x.size())
        out, hn = self.rnn(x, h0.detach())
        # Index hidden state of last time step
        # just want last time step hidden states! 
        out = self.fc(out[:, -1, :])
        return out


def predict_by_sample(model, test_features, test_labels,input_dim, chunk_size, 
                      blocks=100, print_info=False, filename="test"):
    total = 0
    total_patient = 0
    total_control = 0

    acc_total = 0
    acc_total_patient = 0
    acc_total_control = 0

    # calcualte acc by sample individually 
    sample_total = 0
    sample_total_control = 0
    sample_total_patient = 0
    acc_sample = 0
    acc_sample_control = 0
    acc_sample_patient = 0

    plot_labels = []
    chunk_preds = []
    sample_preds = []

    for sample, label in zip(test_features, test_labels):
        key = label
        plot_labels.append(label)

        sample, label = split_chunks([sample], [label], chunk_size)

        samples = [sample[i:i+blocks] for i in range(0,len(sample),blocks)]
        labels = [label[i:i+blocks] for i in range(0,len(label), blocks)]
        
        samples = samples[:-1] # skip the last one since it does not fit nicely as one block 
        labels = labels[:-1] # skip the last one since it does not fit nicely as one block 

        # add part of samples to it 
        chunk_pred = []
        sample_pred = []
        # so we take the chunks that we have and get a prediction per sample 
        # so we take the highest accuracy for the chunks...
        if str(device) == "cuda":
            samples = torch.cuda.FloatTensor(samples)
            labels = torch.cuda.LongTensor(labels)
        else:
            samples = torch.FloatTensor(samples)
            labels = torch.LongTensor(labels)
        
        for sample, label in zip(samples, labels):
            if key == 0:
                sample_total_control += 1
            else:
                sample_total_patient += 1
            sample_total += 1

            #print(samples.size())
            sample = sample.view(blocks, input_dim)
            # Forward pass only to get logits/output
            outputs = model(sample)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # turn it into a list 
            chunk_pred.extend(predicted.data.tolist())
            
            # Total number of labels
            total += label.size()[0]

            # Total correct predictions
            acc_total += (predicted == label).sum()

            if key == 0:
                acc_total_control += (predicted == label).sum()
                total_control += label.size()[0]
            else:
                acc_total_patient += (predicted == label).sum()
                total_patient += label.size()[0]
            # just realized that I just need to take the fifty percent majority 
            flags = (predicted == label).data.tolist()
            predictions = Counter(flags)
            assert sum(predictions.values()) == label.size()[0]
            value = 0
            if predictions[True] > sum(predictions.values())//2:
                if key == 0:
                    acc_sample_control += 1
                    value = 0
                else:
                    acc_sample_patient += 1
                    value = 1
                acc_sample += 1
            # it is the opposite since it is wrong
            else:
                if key == 0:
                    value = 1
                else:
                    value = 0
            #if print_info:
            #    print(predictions, key)
            sample_pred.append(value)
        # we are adding the samples 
        chunk_preds.append(chunk_pred)
        sample_preds.append(sample_pred)
    chunk_total_acc = 100.0*calc_balance_acc(chunk_preds, plot_labels)
    chunk_pd_acc = 100.0*acc_total_patient/total_patient
    chunk_con_acc = 100.0*acc_total_control/total_control
    print("Chunk:")
    print(f"Accuracy: {chunk_total_acc}, " + 
          f"Patient Accuracy: {chunk_pd_acc}, " + 
          f"Control Accuracy: {chunk_con_acc}")

    total_acc = 100.0*calc_balance_acc(sample_preds, plot_labels)
    pd_acc = 100.0*acc_sample_patient/sample_total_patient
    con_acc = 100.0*acc_sample_control/sample_total_control
    print("Sample:")
    print(f"Accuracy: {total_acc}, " + 
          f"Patient Accuracy: {pd_acc}, " + 
          f"Control Accuracy: {con_acc}")

    if filename and print_info:
        # let me plot the accuracies
        start = time.time() 
        plot_output_and_input(plot_labels, chunk_preds, test_features, filename, add_label=None)
        print(f"Total time to plot: {time.time()-start}")
    # I should return the highest accuracy 
    return {"total_acc"       : float(total_acc),
            "pd_acc"          : float(pd_acc),
            "con_acc"         : float(con_acc),
            "chunk_total_acc" : float(chunk_total_acc),
            "chunk_pd_acc"    : float(chunk_pd_acc),
            "chunk_con_acc"   : float(chunk_con_acc)}

def train(filepath, chunk_size, num_epochs, model, print_info, 
          prefix="test", random_state=42):
    start = time.time()
    input_dim = chunk_size
    output_dim = 2
    # change the block size
    # this is very hard coded for now 
    blocks = 100

    data_name = f"chunk_{chunk_size}_{random_state}.npy"

    #if check_file(data_name):
    #    train_features, test_features, train_labels, test_labels = np.load(data_name, allow_pickle=True)
    #else:
    #    train_features, test_features, train_labels, test_labels = create_data(filepath, chunk_size, blocks, random_state)
    #    np.save(f"chunk_{chunk_size}_{random_state}", np.array([train_features, test_features, train_labels, test_labels]))
    train_features, test_features, train_labels, test_labels = create_data(filepath, chunk_size, blocks, random_state)
    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # the plot name for all the plots
    name = ntpath.basename(filepath).split(".")[0]
    plot_name = f"{model.name}_{name}_{prefix}_{chunk_size}_{num_epochs}_{random_state}"

    # so wer are not using this schedular since it does not work 
    #lmbda = lambda epoch: 0.95
    #scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    # Assuming optimizer uses lr = 0.05 for all groups
    # lr = 0.05     if epoch < 30
    # lr = 0.005    if 30 <= epoch < 80
    # lr = 0.0005   if epoch >= 80
    #milestones = [10,20,30,40]
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.25)
    
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    # store the losses
    all_losses = []
    current_loss = 0
    print(f"Number of train: {len(train_labels)}")
    #quit()
    print_class_number(test_labels)

    top_acc = 0
    acc_dict = {}
    top_iter = 0
    # do it for the other accuracy 
    chunk_top_acc = 0
    chunk_acc_dict = {}
    chunk_top_iter = 0

    # store the last ten loss has not improved the scheduling 
    previous_loss = 1
    counter = 10

    # store the best results
    results = {}
    all_accs_chunk = [] 
    all_accs_sample = []

    train_features = torch.Tensor(train_features)
    train_labels = torch.LongTensor(train_labels)

    if str(device) == "cuda":
        print(f"Model is using Cuda: {next(model.parameters()).is_cuda}")
        print(f"Cuda device: {torch.cuda.get_device_name(0)}")
        train_features = train_features.type(torch.cuda.FloatTensor)
        train_labels = train_labels.type(torch.cuda.LongTensor)

    iter = 0
    training_steps = len(train_features)
    print(f"training steps: {training_steps}")
    delta = 0.001

    for epoch in range(num_epochs):
        epoch_start = time.time()
        # reset the loss to zero after each epoch
        total_loss = 0

        for i, (samples, labels) in enumerate(zip(train_features, train_labels)):#train_loader):
            model.train()
            samples = samples.view(blocks,input_dim)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(samples)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            # store the loss 
            total_loss += loss.item()

            if abs(loss.item() - previous_loss) <= delta:
                counter += 1
                if  counter == 10:
                    counter = 0
                    print("\nLOSS HAVE NOT CHANGED FOR 10 STEPS\n")
            previous_loss = loss.item()
            # iter += 1
            # if iter % training_steps == 0:
            #     scheduler.step(previous_loss)

        model.eval()
        filename = None
        # just the first and last
        if print_info and epoch in [0, num_epochs-1]:
            filename = os.path.join("predictions", f"{plot_name}_{epoch}_{chunk_size}")
        
        results = predict_by_sample(model, test_features, test_labels,
                                    input_dim, chunk_size, blocks,
                                    print_info=print_info,
                                    filename=filename)
        
        if results["total_acc"] > top_acc:
            acc_dict = results
            top_acc = results["total_acc"]
            top_iter = epoch

        if results["chunk_total_acc"] > chunk_top_acc:
            chunk_acc_dict = results
            chunk_top_acc = results["chunk_total_acc"]
            chunk_top_iter = epoch

        # append it to the lists 
        all_accs_chunk.append(results["chunk_total_acc"]/100)
        all_accs_sample.append(results["total_acc"]/100)
        
        # Print Loss
        print('Epoch: {}, Loss: {}, Duration: {}'.format(epoch, total_loss/training_steps, time.time()-epoch_start))
        all_losses.append(total_loss/training_steps)
        # step corresponds to the epoch 
        scheduler.step(total_loss/training_steps)

    print(f"Total time: {time.time() - start}")
    chunk_top_name = "Chunk:{0}_{1}_{2}_{3}".format(round(chunk_acc_dict["chunk_total_acc"],2),
                                                round(chunk_acc_dict["chunk_pd_acc"],2),
                                                round(chunk_acc_dict["chunk_con_acc"],2),
                                                chunk_top_iter)
    top_point = [(chunk_top_iter-1, chunk_top_name)]
    top_name = "Sample:{0}_{1}_{2}_{3}".format(round(acc_dict["total_acc"],2),
                                        round(acc_dict["pd_acc"],2),
                                        round(acc_dict["con_acc"],2),
                                        top_iter)
    top_point.append((top_iter-1, top_name))
    minor_freq = 1
    major_freq = 10#num_epochs // minor_freq
    epoch_loss_plot(all_losses,all_accs_chunk, plot_name, top_point,
                    accs_name="Chunk Accuracy", accs2=all_accs_sample, 
                    accs2_name="Sample Accuracy", minor_freq=minor_freq, major_freq=major_freq)

def run(filepath, chunk_size, num_epoches, hidden_dims, print_info, prefix, random_state): 
    seed_torch()
    torch.cuda.empty_cache()
    model = RNNModelMultiLayer(chunk_size, 2, hidden_dims, batch_size=100)
    train(filepath, chunk_size, num_epoches, model, print_info, prefix=prefix, random_state=random_state)

def run_lstm(filepath, chunk_size, num_epoches, hidden_dims, print_info, prefix, random_state): 
    seed_torch()
    torch.cuda.empty_cache()
    model = LSTMModelMultiLayer(chunk_size, 2, hidden_dims, batch_size=100)
    train(filepath, chunk_size, num_epoches, model, print_info, prefix=prefix, random_state=random_state)

def run_plain_rnn(filepath, chunk_size, num_epoches, hidden_dim,layer_dim, print_info, prefix, random_state): 
    seed_torch()
    torch.cuda.empty_cache()
    model = RNNModel(chunk_size, 2, hidden_dim,layer_dim=layer_dim, batch_size=100)

    train(filepath, chunk_size, num_epoches, model, print_info, prefix=prefix, random_state=random_state)

def run_plain_lstm(filepath, chunk_size, num_epoches, hidden_dim,layer_dim, print_info, prefix, random_state): 
    seed_torch()
    from test_lstm import LSTMModel
    model = LSTMModel(chunk_size, 2, hidden_dim,layer_dim=layer_dim, batch_size=100)
    torch.cuda.empty_cache()
    train(filepath, chunk_size, num_epoches, model, print_info, prefix=prefix, random_state=random_state)

if __name__ == "__main__":
    # e.g.
    # filepath = "data/sample_peaks_4.npy"
    # filepath_with_dist = "data/peaks_with_dist_4.npy"
    # filepath_just_dist = "data/dist_4.npy"
    # chunk_size = 10 # need to double it since it has distances 
    # num_layers = 1
    filepath = sys.argv[1] # we should just be using with one with four microstates
    chunk_size = 15
    num_epoches = 100
    #hidden_units = [int(x) for x in sys.argv[4].split(",")]
    hidden_dims = [2048,1024,512]
    prefix = "bad"
    #random_state = 3
    layer_dims = len(hidden_dims)
    print_info = False
    for random_state in range(10):
        run(filepath, chunk_size, num_epoches, hidden_dims, print_info, 
            prefix, random_state)
