#
# interesting but rather useless for my case:
# http://probcomp.csail.mit.edu/blog/programming-and-probability-sampling-from-a-discrete-distribution-over-an-infinite-set/
#

import os
import random
import numpy as np
import scipy.stats as ss
from mpmath import bell
from mpmath import e
from mpmath import factorial
from mpmath import power

# to train the model
from test_rnn import RNNModelMultiLayer
from test_rnn import LSTMModelMultiLayer
from test_rnn import seed_torch
from utils import split_chunks

from sklearn.model_selection import train_test_split
import time 
from sklearn.metrics import balanced_accuracy_score
from utils import print_class_number
from utils import upsample

import torch
import torch.nn as nn
# make sure the cuda works 
from torch.autograd import Variable

from collections import Counter
from utils import epoch_loss_plot
from utils import remove_duplicates

# plot the curves 
import matplotlib.pyplot as plt

import ntpath

def generate_random_seq_weighted(items, p, k, seed=42):
    np.random.seed(seed)
    return np.random.choice(items, k, p=p)

def create_sine(sample_length, num_samples, a=10, b=2):
    print(f"Num samples: {num_samples}")
    t = np.linspace(0, num_samples*sample_length, num=num_samples+sample_length)
    x = a*np.sin(t*b)
    # Generate noise samples
    mean_noise = 0
    noise = np.random.normal(mean_noise, 1, len(x))

    # Noise up the original signal (again) and plot
    y = x + noise
    # need to plot the two functions
    # now we just return the samples from the sample length 
    # Plot signal with noise
    # plt.plot(t[:sample_length*10], y[:sample_length*10])
    # plt.title('Signal with noise')
    # plt.ylabel('Voltage (V)')
    # plt.xlabel('Time (s)')
    # plt.show()
    data = np.array([y[i:i + sample_length] for i in range(0, len(y)-sample_length)])
    print(len(data))
    # print(len(y))
    return data

def create_x_y_sin(num_samples, chunk_size, random_state=42):
    class_A = create_sine(chunk_size, num_samples, 10, 10)
    class_B = create_sine(chunk_size, num_samples, 10, 20)

    Y = np.array([0]*len(class_A) + [1]*len(class_B))
    X = np.append(class_A ,class_B, axis=0)
    train_features, test_features, train_labels, test_labels = train_test_split(X,Y, 
                                                                                test_size = 0.25,
                                                                                random_state = random_state)
    print(train_features.shape)
    data = {}
    data["trf"] = train_features
    data["trl"] = train_labels
    data["tef"] = test_features
    data["tel"] = test_labels
    return data


# they have the same random distribution at length four along the sequence 
def generate_based_on_random_distribution(num_samples, seq_length, contigious_length=4, scale=3):
    # do set a random seed since we want to shift the range of indices
    # we have the same normal distribution from 0 -> 1 
    # it is hard coded since there are all four states 
    x = np.arange(0, 4)
    xU, xL = x + 0.5, x - 0.5 
    prob = ss.norm.cdf(xU, scale = scale) - ss.norm.cdf(xL, scale = scale)
    prob = prob / prob.sum() #normalize the probabilities so their sum is 1
    # we have the same distribution for blocks of 4 so it is greater than 
    samples = []
    print(prob)
    p = [1/len(x)]*len(x)
    for _ in range(num_samples):
        sequence = generate_random_seq_weighted(x, p, seq_length)
        index = random.randint(0,seq_length-contigious_length)
        part_seq = generate_random_seq_weighted(x, prob, contigious_length)
        sequence[index:index+contigious_length] = part_seq
        samples.append(sequence)
    return np.array(samples)

def create_random_X_Y(num_samples, chunk_size, partial_size1, partial_size2):
    class_A = generate_based_on_random_distribution(num_samples, chunk_size, 
                                                    contigious_length=partial_size1,
                                                    scale=2)
    class_B = generate_based_on_random_distribution(num_samples, chunk_size, 
                                                    contigious_length=partial_size2,
                                                    scale=3)
    Y = np.array([0]*num_samples + [1]*num_samples)
    X = np.append(class_A ,class_B, axis=0)
    return X,Y

def create_x_y_diff_partial(num_samples, chunk_size, partial_size1, partial_size2, random_state=42):
    X,Y= create_random_X_Y(num_samples, chunk_size, partial_size1, partial_size2)
    train_features, test_features, train_labels, test_labels = train_test_split(X,Y, 
                                                                                test_size = 0.25,
                                                                                random_state = random_state)
    data = {}
    data["trf"] = train_features
    data["trl"] = train_labels
    data["tef"] = test_features
    data["tel"] = test_labels
    return data

# generate synthetic samples for x and y
def create_x_y(num_samples, chunk_size, partial_size, random_state=42):
    class_A = np.array([generate_random_seq_weighted([x for x in range(4)], [0.25]*4, chunk_size) for _ in range(num_samples)])
    class_B = generate_based_on_random_distribution(num_samples, chunk_size, contigious_length=partial_size)
    Y = np.array([0]*num_samples + [1]*num_samples)
    X = np.append(class_A ,class_B, axis=0)
    train_features, test_features, train_labels, test_labels = train_test_split(X,Y, 
                                                                                test_size = 0.25,
                                                                                random_state = random_state)
    data = {}
    data["trf"] = train_features
    data["trl"] = train_labels
    data["tef"] = test_features
    data["tel"] = test_labels
    return data


SIZES_CONTROL = 24
SIZES_DISEASE = 44

def create_x_y_from_file_and_from_random(filepath,chunk_size, blocks, disease_first=True,
                                test_size=0.1, random_state=42, get_rid_dups=False,
                                skip=0):
    # we split based on the samples but classify based on the chunks 
    data = np.load(filepath, allow_pickle=True)
    print(data.shape)

    data_only_disease = []
    if disease_first:
        for i in range(SIZES_DISEASE):
            data_only_disease.append(data[i])
    else:
        for i in range(SIZES_CONTROL,SIZES_CONTROL+SIZES_DISEASE):
            data_only_disease.append(data[i])
    
    # we split it in half
    assert len(data_only_disease) % 2 == 0

    labels = [0]*SIZES_DISEASE

    for i in range(0,len(data_only_disease),2):
        data_only_disease[i] = list(generate_random_seq_weighted([x for x in range(4)], 
                                                            [0.25]*4,
                                                            k = len(data_only_disease[i])))
        labels[i] = 1

    return load_data(data_only_disease, labels, test_size, random_state, get_rid_dups, skip)

# 
# we take the disease group and split it in half and randomize the sequence data and pass it 
# it in as a file 
def create_x_y_from_file_random(filepath,chunk_size, blocks, disease_first=True,
                                test_size=0.1, random_state=42, get_rid_dups=False,
                                skip=0):
    # we split based on the samples but classify based on the chunks 
    data = np.load(filepath, allow_pickle=True)
    print(data.shape)

    data_only_disease = []
    if disease_first:
        for i in range(SIZES_DISEASE):
            data_only_disease.append(data[i])
    else:
        for i in range(SIZES_CONTROL,SIZES_CONTROL+SIZES_DISEASE):
            data_only_disease.append(data[i])
    
    # we split it in half
    assert len(data_only_disease) % 2 == 0

    for i in range(len(data_only_disease)//2):
        print(data_only_disease[i][0:10])
        random.shuffle(data_only_disease[i])
        print(data_only_disease[i][0:10])

    half = SIZES_DISEASE//2
    labels = [0]*half + [1]*half

    return load_data(data_only_disease, labels, test_size, random_state, get_rid_dups, skip)

def create_x_y_from_file(filepath, chunk_size, blocks, disease_first=True, 
                         test_size=0.1, random_state=42, get_rid_dups=False,
                         skip=1):
    # we split based on the samples but classify based on the chunks 
    data = np.load(filepath, allow_pickle=True)
    print(data.shape)

    labels = [0]*SIZES_CONTROL+[1]*SIZES_DISEASE
    if disease_first:
        labels = [1]*SIZES_DISEASE+[0]*SIZES_CONTROL
    print(f"SKIP:{skip}")
    return load_data(data, labels, test_size, random_state, get_rid_dups, skip)

def load_data(data, labels, test_size, random_state, get_rid_dups, skip):
    # I assure that the 
    train_features, test_features, train_labels, test_labels = train_test_split(data,labels, 
                                                               test_size = test_size,
                                                               random_state = random_state)
    print("Sample Level:")
    class_one_count, class_two_count = print_class_number(test_labels)
    if class_one_count == 0 or class_two_count == 0:
        raise ValueError(f"The number of class one: {class_one_count} and \
                           class two: {class_two_count} is invalid for testing")

    train_features, train_labels = split_chunks(train_features, train_labels, chunk_size, skip=skip)
    test_features, test_labels = split_chunks(test_features, test_labels, chunk_size, skip=skip)

    # get rid of chunks that are the same 
    if get_rid_dups:
        print("Analyze similarities")
        train_features, train_labels, test_features, test_labels = remove_duplicates(train_features, 
                                                                                     train_labels, 
                                                                                     test_features, 
                                                                                     test_labels)

    print("Chunk Level:")
    print_class_number(train_labels)
    # we need to upsample the train_features 
    train_features, train_labels = upsample(np.array(train_features), np.array(train_labels), random_state)
    print("Chunk Level Upsample:")
    print_class_number(train_labels)

    print("Before:")
    print(f"Chunks Train: {len(train_labels)}")
    print(f"Chunks Test: {len(test_labels)}")

    # I have to block the data so if the length is not the number of blocks 
    possible_blocks_train = len(train_features)//blocks * blocks
    train_features = train_features[:possible_blocks_train,]
    train_labels = train_labels[:possible_blocks_train]

    possible_blocks_test = len(test_labels)//blocks * blocks
    test_features = test_features[:possible_blocks_test]
    test_labels = test_labels[:possible_blocks_test]

    print("After:")
    print(f"Chunks Train: {len(train_labels)}")
    print(f"Chunks Test: {len(test_labels)}")

    data = {}
    data["trf"] = train_features
    data["trl"] = train_labels
    data["tef"] = test_features
    data["tel"] = test_labels
    return data

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return (TP, FP, TN, FN)

def predict(model, X,Y):
    true_labels = []
    preds = []

    for samples, labels in zip(X, Y):
        # Forward pass only to get logits/output
        outputs = model(samples)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # turn it into a list 
        preds.extend(predicted.data.tolist())
        true_labels.extend(labels.data.tolist())
    total_acc = 100.0*balanced_accuracy_score(true_labels, preds)
    TP, FP, TN, FN = perf_measure(true_labels, preds)

    class_counts = Counter(true_labels)
    print(f"Balanced Accuracy: {total_acc}, " + 
          f"True Positives: {TP}, False Positives:{FP}, True negatives:{TN}, False negatives:{FN}")

    # I should return the highest accuracy 
    return {"total_acc"          : float(total_acc),
            "partial_random_acc" : float(TP/class_counts[1]),
            "random_acc"         : float(TN/class_counts[0])}


def shuffle(X,Y):
    perm = np.random.permutation(len(X))
    return X[perm], Y[perm]

def train(data, num_epochs, model, filename, blocks=100):
    start = time.time()
    output_dim = 2
    
    train_features = data["trf"]
    train_labels = data["trl"]
    test_features = data["tef"]
    test_labels = data["tel"]

    input_dim = train_features.shape[1]
    
    train_features = np.array([train_features[i:i+blocks] for i in range(0,len(train_features),blocks)])
    test_features = np.array([test_features[i:i+blocks] for i in range(0,len(test_features),blocks)])
    train_labels = np.array([train_labels[i:i+blocks] for i in range(0,len(train_labels),blocks)])
    test_labels = np.array([test_labels[i:i+blocks] for i in range(0,len(test_labels),blocks)])

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    # store the losses
    all_losses = []
    current_loss = 0
    training_steps = len(train_features)
    print(f"Number of train: {training_steps}")

    # store the best test accuracy
    top_acc = 0
    acc_dict = {}
    top_iter = 0

    # store all the accuracies
    all_accs_test = []
    all_accs_train = []

    # store the last ten loss has not improved the scheduling 
    previous_loss = 1
    counter = 10

    train_features = torch.Tensor(train_features)
    train_labels = torch.LongTensor(train_labels)
    test_features = torch.Tensor(test_features)
    test_labels = torch.LongTensor(test_labels)

    train_features = train_features.type(torch.cuda.FloatTensor)
    train_labels = train_labels.type(torch.cuda.LongTensor)
    test_features = test_features.type(torch.cuda.FloatTensor)
    test_labels = test_labels.type(torch.cuda.LongTensor)

    # shuffle after each epoch 

    for epoch in range(num_epochs):
        epoch_start = time.time()
        # reset the loss to zero after each epoch
        total_loss = 0

        train_features, train_labels = shuffle(train_features, train_labels)

        for i, (samples, labels) in enumerate(zip(train_features, train_labels)):
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
            previous_loss = loss.item()

        model.eval()
        results_train = predict(model, train_features, train_labels)
        results_test = predict(model, test_features, test_labels)
        
        if results_test["total_acc"] > top_acc:
            acc_dict = results_test
            top_acc = results_test["total_acc"]
            top_iter = epoch

        # append it to the lists 
        all_accs_test.append(results_test["total_acc"]/100)
        all_accs_train.append(results_train["total_acc"]/100)
        
        # Print Loss
        print('Epoch: {}, Loss: {}, Duration: {}'.format(epoch, total_loss/training_steps, time.time()-epoch_start))
        all_losses.append(total_loss/training_steps)
        # step corresponds to the epoch 
        scheduler.step(total_loss/training_steps)

    print(f"Total time: {time.time() - start}")
    top_name = "Chunk:{0}_{1}_{2}_{3}".format(round(acc_dict["total_acc"],2),
                                              round(acc_dict["partial_random_acc"],2),
                                              round(acc_dict["random_acc"],2),
                                              top_iter)
    top_point = [(top_iter-1, top_name)]

    minor_freq = 1
    major_freq = 10
    epoch_loss_plot(all_losses,all_accs_test, filename, top_point,
                    accs_name="Test Accuracy", accs2=all_accs_train, 
                    accs2_name="Train Accuracy", minor_freq=minor_freq, major_freq=major_freq)


# tests I have done:
# <hidden dims> : <hightest accuracy> <dropout>
# this is for chunk size of ten with partial of 4 randomly in each sequence
# [128] : 92% 0.1
# [256] : 92.65% 0.1
# [512] : 92.75% 0.1 -> converges really early at 10 epochs
# [1024] : 92.89% 0.1 -> converges really early at 6 epochs
# [1024,512] : 92.19% 0.1 -> converges later at 79 epochs
#
# BEST
# [1024,1024] : 92.97% 0.1 
# 
# [1024,1024,1024] : 92.71% 0.2
# [1024,1024] : 92.07% 0.2
# [1024,512,256] : 90.21% 0.1 -> converges later at 97 epochs
# [1024,512,256] : 90.95% 0.1 -> converges later at 356 epochs
# [256] : 92.45% 0.2
# [128, 64] : 88% 0.1 
# [128, 64] : 74% 0.5 

# shuffle randomly 
# so we shuffle sort all the sequences just get letter 4 chunks 
class ModelMultiLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_size=100, dropout=0.10, model_name="RNN"):
        # we are just taking the first RNN 
        super(ModelMultiLayer, self).__init__()
        self.model_name = model_name
        self.rnns = []
        self.vars = []
        self.biases = []
        self.dropouts = []
        input_dim = input_dim
        p = dropout
        for index, hidden_dim in enumerate(hidden_dims):
            if model_name == "RNN":
                self.rnns.append(nn.RNN(input_dim, hidden_dim, 1, batch_first=True).cuda())
                self.vars.append(Variable(torch.zeros(1, batch_size, hidden_dim).type(torch.cuda.FloatTensor), 
                             requires_grad=True))
            elif model_name == "BiRNN":
                self.rnns.append(nn.RNN(input_dim, hidden_dim, 1, batch_first=True, bidirectional=True).cuda())
                self.vars.append(Variable(torch.zeros(2, batch_size, hidden_dim).type(torch.cuda.FloatTensor), 
                             requires_grad=True))
            elif model_name == "LSTM":
                self.rnns.append(nn.LSTM(input_dim, hidden_dim, 1, batch_first=True).cuda())
                self.vars.append(Variable(torch.zeros(1, batch_size, hidden_dim).type(torch.cuda.FloatTensor), 
                             requires_grad=True))
                self.biases.append(Variable(torch.zeros(1, batch_size, hidden_dim).type(torch.cuda.FloatTensor), 
                             requires_grad=True))
            elif model_name == "BiLSTM":
                self.rnns.append(nn.LSTM(input_dim, hidden_dim, 1, batch_first=True, bidirectional=True).cuda())
                self.vars.append(Variable(torch.zeros(2, batch_size, hidden_dim).type(torch.cuda.FloatTensor), 
                             requires_grad=True))
                self.biases.append(Variable(torch.zeros(2, batch_size, hidden_dim).type(torch.cuda.FloatTensor), 
                             requires_grad=True))
            else:
                raise TypeError(f"The model name {model_name} is not valid should be RNN, BiRNN or LSTM") 
            
            self.dropouts.append(nn.Dropout(p=p))
            print(p)
            p /= (2*(index+1))
            input_dim = hidden_dim
            if model_name == "BiRNN" or model_name == "BiLSTM":
                input_dim *= 2
        if model_name == "BiRNN" or model_name == "BiLSTM":
            self.fc = nn.Linear(input_dim, output_dim).cuda()
        else:
            self.fc = nn.Linear(input_dim, output_dim).cuda()
        hidden_dims_name = "_".join([str(x) for x in hidden_dims])
        # add a name for print reasons
        self.name = f"{model_name}_MultiLayer_{hidden_dims_name}_{int(dropout*100)}"

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        out = x.view(x.size(0),1,x.size(1))
        for index, (var, rnn) in enumerate(zip(self.vars, self.rnns)):
            if self.model_name == "LSTM":
                c  = self.biases[index]
                out, (hn, cn) = rnn(out, (var, c))
            elif self.model_name == "RNN":
                out, hn = rnn(out, var.detach())
            elif self.model_name == "BiRNN":
                bi_output, bi_hidden = rnn(out, var.detach())
                hidden_size = var.size()[2]
                forward_output, backward_output = bi_output[:, :, :hidden_size], bi_output[:, :, hidden_size:]
                out = torch.cat((forward_output, backward_output), dim=-1)
            elif self.model_name == "BiLSTM":
                c  = self.biases[index]
                bi_output, (bi_hidden, cn) = rnn(out, (var, c))
                hidden_size = var.size()[2]
                forward_output, backward_output = bi_output[:, :, :hidden_size], bi_output[:, :, hidden_size:]
                out = torch.cat((forward_output, backward_output), dim=-1)
            out = self.dropouts[index](out)
        # Index hidden state of last time step
        # just want last time step hidden states!
        return self.fc(out[:, -1, :])


if __name__ == "__main__":
    chunk_size = 150
    blocks = 100
    hidden_dims = [2048,1024,512]
    output_dim = 2
    num_epochs = 50

    hidden_dims_name = ",".join([str(x) for x in hidden_dims])

    model_name = "BiLSTM"
    #testing_method = "artificial"
    #random_state=42
    filepath = "sample_peaks_5.npy"
    disease_first = False
    skip = 1
    testing_method = "data"
    for random_state in range(10):
        if testing_method == "data":
            head, tail = ntpath.split(filepath)
            filepart = tail.split(".npy")[0]
            filename = f"TEST_{testing_method}_{model_name}_{chunk_size}_{skip}_{num_epochs}_{hidden_dims_name}_{random_state}_{filepart}"
            filename = os.path.join(head, filename)
        else:
            filename = f"TEST_{testing_method}_{model_name}_{chunk_size}_{num_epochs}_{hidden_dims_name}_{random_state}"
            filename = os.path.join("D:\\pd_data", filename)

        if testing_method == "artificial":
            num_samples = 10000
            partial_size = 3
            data = create_x_y(num_samples, chunk_size, partial_size=partial_size)
        elif testing_method == "artificial2":
            num_samples = 10000
            partial_size1 = 4
            partial_size2 = 4
            data = create_x_y_diff_partial(num_samples, chunk_size, partial_size1, partial_size2)
        elif testing_method == "random_w_data":
            data = create_x_y_from_file_and_from_random(filepath,chunk_size,blocks, 
                                                        skip=skip, random_state=random_state, test_size=0.2,
                                                        disease_first=disease_first)
        elif testing_method == "data_only_disease_shuffled_half":
            data = create_x_y_from_file_random(filepath,chunk_size,blocks, skip=skip, 
                                            random_state=random_state, test_size=0.2,
                                            disease_first=disease_first)
        elif testing_method == "sin":
            num_samples = 10000
            data = create_x_y_sin(num_samples, chunk_size)
        elif testing_method == "data":
            data = create_x_y_from_file(filepath,chunk_size,blocks, skip=1, 
                                        random_state=random_state, test_size=0.2,
                                        disease_first=disease_first)
        else:
            raise KeyError("Invalid testing method chosen")
        seed_torch()
        torch.cuda.empty_cache()
        model = ModelMultiLayer(chunk_size, output_dim, hidden_dims, batch_size=blocks, dropout=0.1, model_name = model_name)
        train(data, num_epochs, model, filename)
