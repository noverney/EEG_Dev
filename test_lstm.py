#
# Author: Normand Overney
#

import torch
import torch.nn as nn
import time
import numpy as np
import os

# use sklearn to split the data 
from sklearn.model_selection import train_test_split

from utils import sizes_control, sizes_disease
from utils import plot_lists
import random
from utils import upsample
from utils import calc_balance_acc
from utils import epoch_loss_plot

import sys
# get counter 
from collections import Counter
import ntpath

# imbalance 
from sklearn.metrics import balanced_accuracy_score

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

# set up the cuda version
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(f"Pytorch device: {device}")

def predict_by_sample(model, test_features, test_labels,input_dim):
    # calcualte acc by sample individually 
    sample_total = len(test_features)
    sample_total_control = 0
    sample_total_patient = 0
    acc_sample_control = 0
    acc_sample_patient = 0

    labels = test_labels

    test_features = torch.Tensor(test_features)
    test_labels = torch.LongTensor(test_labels)

    if str(device) == "cuda":
        test_features = test_features.type(torch.cuda.FloatTensor)
        test_labels = test_labels.type(torch.cuda.LongTensor)

    sample_preds = []
    for sample, label in zip(test_features, test_labels):
        key = label

        if key == 0:
            sample_total_control += 1
        else:
            sample_total_patient += 1
        
        #label = np.array([label])
        label = label.view(1)
        sample = sample.view(-1, input_dim)

        #print(samples.size())
        #print(labels.size())
        
        # Forward pass only to get logits/output
        outputs = model(sample)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # just realized that I just need to take the fifty percent majority 
        flags = (predicted == label).data.tolist()
        if flags[0]:
            if key == 0:
                acc_sample_control += 1
            else:
                acc_sample_patient += 1

        sample_preds.append(predicted.data.tolist()[0])

    total_acc = 100.0*balanced_accuracy_score(labels, sample_preds)
    pd_acc = 100.0*acc_sample_patient/sample_total_patient
    con_acc = 100.0*acc_sample_control/sample_total_control
    print(f"Accuracy: {total_acc}, " + 
          f"Patient Accuracy: {pd_acc}, " + 
          f"Control Accuracy: {con_acc}")
    # I should return the highest accuracy 
    return {"total_acc"       : float(total_acc),
            "pd_acc"          : float(pd_acc),
            "con_acc"         : float(con_acc)}

# I could chunk this 1000 but that is the old version of the model 
# but then that was not working so well instead of passing the full sample 
def make_same_lengths(data):
    new_data = []
    largest = 0
    for sample in data:
        if len(sample) > largest:
            largest = len(sample)
    print(f"Largest Length of samples: {largest}")
    for sample in data:
        #new_sample = sample + [-1]*(largest-len(sample))
        new_sample = np.concatenate((sample, [-1]*(largest-len(sample))), axis=None)
        new_data.append(new_sample)
    new_data = np.array(new_data)
    print(new_data.shape)
    return new_data

def create_data(filepath,random_state):
    data = np.load(filepath, allow_pickle=True)
    print(data.shape)

    # fill the rest of the lenghts with zero 
    data = make_same_lengths(data)

    labels = [0]*len(sizes_control)+[1]*len(sizes_disease)
    test_size = 0.20
    random_state = 42
    train_features, test_features, train_labels, test_labels = train_test_split(data,labels, 
                                                               test_size = test_size,
                                                               random_state = random_state)

    start = time.time()
    train_features, train_labels = upsample(np.array(train_features), 
                                            np.array(train_labels), random_state)
    print(f"Clustering time: {time.time()-start}")
    # we need to randomly shuffle the lists 
    combined = list(zip(train_features, train_labels))
    random.shuffle(combined)
    train_features, train_labels = zip(*combined)
    # probably should use some up-sampling or synthetic sampling 
    print(f"Number of Samples: {len(train_labels)}")

    return train_features, test_features, train_labels, test_labels

# training need to get sample to full length 
def train(filepath, num_epochs, CreateModel, hidden_units, prefix, random_state):
    start = time.time()
    train_features, test_features, train_labels, test_labels = create_data(filepath, random_state)
    input_dim = len(train_features[0])
    output_dim = 2

    # change the block size
    model = CreateModel(input_dim, output_dim, hidden_units)
    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.0001
    # essentially learning_rate*weight_decay
    # weight_decay = 1

    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # store the losses
    all_losses = []
    current_loss = 0

    print(f"Number of training: {len(train_features)}")

    top_acc = 0
    acc_dict = {}
    top_iter = 0

    # store the last ten loss has not improved the scheduling 
    previous_loss = 1
    counter = 10

    # store the current accuracy not increase 
    current_acc = 0
    second_time = False
    results = {}
    all_accs = []

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
        total_loss = 0
        for i, (images, labels) in enumerate(zip(train_features, train_labels)):
            model.train()
            # Load images as tensors with gradient accumulation abilities
            #labels = np.array([labels])
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            # outputs.size() --> 100, 10
            images = images.view(-1,input_dim).requires_grad_()
            labels = labels.view(1)

            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            #print(outputs.size())
            #print(labels.size())
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

        model.eval()
        results = predict_by_sample(model, test_features, test_labels, input_dim)
        current_acc = results["total_acc"]
        if current_acc > top_acc:
            acc_dict = results
            top_acc = current_acc
            top_iter = epoch

        # print accs 
        all_accs.append(current_acc/100)

        # Print Loss
        print('Epoch: {}. Loss: {}'.format(epoch, loss.item()))
        all_losses.append(total_loss/training_steps)

        scheduler.step(total_loss/training_steps)

        #print(f"Epoch {epoch}: {time.time() - epoch_start}")

    print(f"Total time: {time.time() - start}")
    top_name = "Sample:{0}_{1}_{2}".format(round(acc_dict["total_acc"],2),
                                           round(acc_dict["pd_acc"],2),
                                           round(acc_dict["con_acc"],2))
    top_point = [(top_iter-1, top_name)]
    name = ntpath.basename(filepath).split(".")[0]
    plot_name = f"{model.name}_{name}_{prefix}_{num_epochs}_{random_state}"
    
    minor_freq = 1
    major_freq = 10
    epoch_loss_plot(all_losses,all_accs, plot_name, top_point, 
                    minor_freq=minor_freq, major_freq=major_freq)
    return top_acc

# same model as in test_rnn but I have not yet moved that method 
class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_size=1, dropout=0.10):
        # we are just taking the first RNN 
        super(LSTMModel, self).__init__()

        self.lstms = []
        self.vars = []
        self.biases = []
        self.dropouts = []
        input_dim = input_dim
        p = dropout
        for index, hidden_dim in enumerate(hidden_dims):
            self.lstms.append(nn.LSTM(input_dim, hidden_dim, 1).cuda())
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
        self.name = f"LSTMModel_{hidden_dims_name}_{int(dropout*100)}"

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

def run(filepath, num_epoches, model, hidden_units, prefix, random_state):
    seed_torch()
    torch.cuda.empty_cache()
    train(filepath, num_epoches, model, hidden_units, prefix, random_state)

if __name__ == "__main__":
    filepath = "sample_peaks_4.npy"
    num_epoches = 500
    hidden_units = [2048,512]
    prefix = "test"
    random_state = 3
    model = LSTMModel
    run(filepath, num_epoches, model, hidden_units, prefix, random_state)
