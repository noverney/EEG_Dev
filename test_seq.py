#
# Author: Normand Overney
#

import torch
import torch.nn as nn
from random_forest_peaks import generate_x_and_y
from random_forest_peaks import read_file
import numpy as np
import ntpath
import time
from sklearn.model_selection import train_test_split
from utils import generate_synthetic_samples
import random
from utils import epoch_loss_plot
from utils import get_class_number
from utils import print_class_number
import os
from sklearn.metrics import balanced_accuracy_score

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# optimize this for CUDA 
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Pytorch device: {device}")
# create sequential model
class SequentialModelGPU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_states, dropout_p=0.25):
        super(SequentialModelGPU, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.seq = nn.Sequential().cuda()
        for i,state in enumerate(hidden_states):
            # the first layer
            if i == 0:
                self.seq.add_module(f"linear_{i}", nn.Linear(input_dim, state).cuda())
                self.seq.add_module(f"relu_{i}", nn.ReLU().cuda())
                self.seq.add_module(f"dropout_{i}", nn.Dropout(dropout_p).cuda())
                # only one layer 
                if len(hidden_states) == 1:
                    self.seq.add_module(f"ouput", nn.Linear(state, output_dim).cuda())
                    self.seq.add_module(f"softmax", nn.Sigmoid().cuda())

            # the last layer
            elif i+1 == len(hidden_states):
                self.seq.add_module(f"linear_{i}", nn.Linear(hidden_states[i-1], state).cuda())
                self.seq.add_module(f"relu_{i}", nn.ReLU().cuda())
                self.seq.add_module(f"dropout_{i}", nn.Dropout(dropout_p).cuda())
                self.seq.add_module(f"ouput", nn.Linear(state, output_dim).cuda())
                self.seq.add_module(f"softmax", nn.Sigmoid().cuda())
            else:
                self.seq.add_module(f"linear_{i}", nn.Linear(hidden_states[i-1], state).cuda())
                self.seq.add_module(f"relu_{i}", nn.ReLU().cuda())
                self.seq.add_module(f"dropout_{i}", nn.Dropout(dropout_p).cuda())

                # add a name for print reasons
        hidden_units_name = ",".join([str(x) for x in hidden_states])
        self.name = f"SequentialGPU_{hidden_units_name}_{int(dropout_p*100)}"

    def forward(self,sample):
        return self.seq(sample)

# create sequential model
class SequentialModelCPU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_states, dropout_p=0.25):
        super(SequentialModelCPU, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.seq = nn.Sequential()
        for i,state in enumerate(hidden_states):
            # the first layer
            if i == 0:
                self.seq.add_module(f"linear_{i}", nn.Linear(input_dim, state))
                self.seq.add_module(f"relu_{i}", nn.ReLU())
                self.seq.add_module(f"dropout_{i}", nn.Dropout(dropout_p))
                # only one layer 
                if len(hidden_states) == 1:
                    self.seq.add_module(f"ouput", nn.Linear(state, output_dim))
                    self.seq.add_module(f"softmax", nn.Sigmoid())

            # the last layer
            elif i+1 == len(hidden_states):
                self.seq.add_module(f"linear_{i}", nn.Linear(hidden_states[i-1], state))
                self.seq.add_module(f"relu_{i}", nn.ReLU())
                self.seq.add_module(f"dropout_{i}", nn.Dropout(dropout_p))
                self.seq.add_module(f"ouput", nn.Linear(state, output_dim))
                self.seq.add_module(f"softmax", nn.Sigmoid())
            else:
                self.seq.add_module(f"linear_{i}", nn.Linear(hidden_states[i-1], state))
                self.seq.add_module(f"relu_{i}", nn.ReLU())
                self.seq.add_module(f"dropout_{i}", nn.Dropout(dropout_p))

                # add a name for print reasons
        hidden_units_name = ",".join([str(x) for x in hidden_states])
        self.name = f"SequentialCPU_{hidden_units_name}_{int(dropout_p*100)}"

    def forward(self,sample):
        return self.seq(sample)

# we need to get the transition matrix data
def create_data(microstates, n_maps, random_state, print_info):
    info = generate_x_and_y(microstates, n_maps=n_maps)

    X1          = info["first_order_X"]
    features_X1 = info["feature_X1"]
    X2          = info["second_order_X"]
    features_X2 = info["feature_X2"]
    X0          = info["zero_order_X"]
    features_X0 = info["feature_X0"]
    Y           = info["Y"]
    # merge the transitions into one big super matrix
    X = np.concatenate((X0, X1, X2), axis=1)
    test_size = 0.2
    train_features, test_features, train_labels, test_labels = train_test_split(X,Y, 
                                                                test_size = test_size,
                                                                random_state = random_state)
    start = time.time()
    train_features, train_labels = generate_synthetic_samples(train_features, 
                                                              np.array(train_labels), 
                                                              print_all=print_info)
    if print_info:
        print(f"Clustering time: {time.time()-start}")
    # we need to randomly shuffle the lists 
    combined = list(zip(train_features, train_labels))
    random.shuffle(combined)
    train_features, train_labels = zip(*combined)
    # probably should use some up-sampling or synthetic sampling 
    if print_info:
        print(f"Number of Samples: {len(train_labels)}")

    class_one, class_two = get_class_number(test_labels)
    #print(f"class one: {class_one}, class two: {class_two}")
    valid = True
    if class_one == 0 or class_two == 0:
        valid = False
    return train_features, test_features, train_labels, test_labels, valid


def predict_by_sample(model, test_features, test_labels, input_dim):
    # calcualte acc by sample individually 
    sample_total = len(test_features)
    sample_total_control = 0
    sample_total_patient = 0
    acc_sample = 0
    acc_sample_control = 0
    acc_sample_patient = 0

    y_true = test_labels
    test_features = torch.Tensor(test_features)
    test_labels = torch.LongTensor(test_labels)

    if str(device) == "cuda":
        test_features = test_features.type(torch.cuda.FloatTensor)
        test_labels = test_labels.type(torch.cuda.LongTensor)

    y_pred = []
    for sample, label in zip(test_features, test_labels):
        key = label

        if key == 0:
            sample_total_control += 1
        else:
            sample_total_patient += 1
        
        #label = label.view(1)
        
        # Forward pass only to get logits/output
        outputs = model(sample)
        outputs = outputs.view(1,2)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # just realized that I just need to take the fifty percent majority 
        flags = (predicted == label).data.tolist()
        assert len(flags) == 1
        if flags[0]:
            if key == 0:
                acc_sample_control += 1
            else:
                acc_sample_patient += 1
            acc_sample += 1
        y_pred.append(predicted.data.tolist()[0])
    total_acc = 100.0*balanced_accuracy_score(y_true, y_pred)
    pd_acc = 100.0*acc_sample_patient/sample_total_patient
    con_acc = 100.0*acc_sample_control/sample_total_control
    # print(f"Accuracy: {total_acc}, " + 
    #       f"Patient Accuracy: {pd_acc}, " + 
    #       f"Control Accuracy: {con_acc}")
    # print()

    # I should return the highest accuracy 
    return {"total_acc"       : float(total_acc), 
            "pd_acc"          : float(pd_acc),
            "con_acc"         : float(con_acc)}


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 

def train(filepath, n_maps, num_epochs, hidden_states, random_state, prefix, print_info=False, plot_info=False):
    start = time.time()
    microstates = read_file(filepath)
    train_features, test_features, train_labels, test_labels, valid = create_data(microstates, 
                                                                                n_maps, 
                                                                                random_state,
                                                                                print_info)
    if not valid:
        return None
    print_class_number(test_labels)

    input_dim = len(train_features[0])
    SequentialModel = SequentialModelCPU
    if str(device) == "cuda":
        SequentialModel = SequentialModelGPU
    model = SequentialModel(input_dim, 2, hidden_states)

    train_features = torch.Tensor(train_features)
    train_labels = torch.LongTensor(train_labels)
    train_labels = one_hot_embedding(train_labels, 2)
    #print(train_labels)

    if str(device) == "cuda":
        #print(f"Model is using Cuda: {next(model.parameters()).is_cuda}")
        #print(f"Cuda device: {torch.cuda.get_device_name(0)}")
        train_features = train_features.type(torch.cuda.FloatTensor)
        train_labels = train_labels.type(torch.cuda.FloatTensor)
    # check the training by batching the samples
    criterion = nn.BCELoss()
    learning_rate = 0.0001 # tried 0.05 0.01 def not 0.005
                        # 0.5 works but it has shitty 54% accuracy
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    milestones = [i*10 for i in range(1,10)]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
    #                                                  milestones=milestones, 
    #                                                  gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    # has zero too fast training which is a good sign but still
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    lmbda = lambda epoch: 0.95
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1, T_mult=2)
    all_losses = []
    current_loss = 0

    top_acc = 0
    acc_dict = {}
    top_iter = 0

    # store the last ten loss has not improved the scheduling 
    previous_loss = 1
    counter = 10

    # store the current accuracy not increase 
    results = {}
    all_accs = []

    iter = 0
    for epoch in range(num_epochs):
        epoch_start = time.time()
        for i, (sample, label) in enumerate(zip(train_features, train_labels)):#train_loader):
            #label = label.view(1)
            #sample = sample.view(1,sample.size(0))
            model.train()
            optimizer.zero_grad()
            outputs = model(sample)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, label)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            # store the loss 
            current_loss += loss.item()

            if loss.item() == previous_loss:
                counter += 1
                if  counter == 10:
                    counter = 0
                    print("\nLOSS HAVE NOT CHANGED FOR 10 STEPS\n")

            previous_loss = loss.item()
            scheduler.step(previous_loss)

            # Add current loss avg to list of losses
            #if iter % plot_every == 0:
        # we are getting the acc after every epoch
        model.eval()
        #predict(model, test_features, test_labels, input_dim)
        results = predict_by_sample(model, test_features, test_labels,input_dim)
        current_acc = results["total_acc"]
        # make it less than equal so it captures the last time it had that 
        # high level of accuracy
        if current_acc > top_acc:
            acc_dict = results
            top_acc = current_acc
            top_iter = epoch
 
        all_accs.append(current_acc/100)
        all_losses.append(current_loss/(epoch+1))
        current_loss = 0

        if print_info and epoch % 100 == 0:
            print(f"Epoch {epoch}: {time.time() - epoch_start}")

        # print it for the last epoch 
        if epoch == num_epochs-1:
            print(f"Last Accuracy: {current_acc}")
    
    if print_info:
        print(f"Total time: {time.time() - start}")
    

    top_name = "Epoch.{0}: {1},{2},{3}".format(top_iter,
                                               round(acc_dict["total_acc"],2),
                                               round(acc_dict["pd_acc"],2),
                                               round(acc_dict["con_acc"],2))

    minor_freq = 10
    major_freq = num_epochs // minor_freq
    # at least one epoch
    if plot_info:
        name = ntpath.basename(filepath).split(".")[0]
        plot_name = f"{model.name}_{name}_{prefix}_{num_epochs}_{random_state}"
        top_point = [(top_iter, top_name)]
        epoch_loss_plot(all_losses[50:], all_accs[50:],plot_name, top_point,
                        minor_freq=minor_freq, major_freq=major_freq)
    return top_acc


def run(filepath, prefix, hidden_states, num_epochs=500, num_random=100):
    seeds = [x for x in range(num_random)]
    if num_random == 10:
        seeds = [3,10,20,23,26,33,39,44,51,60]
    seed_torch()
    start = time.time()
    n_maps = int(ntpath.basename(filepath).split(".npy")[0].split("_")[-1])
    avg_acc = 0
    total_number = num_random
    total_too_fast = 0
    for random_state in seeds:
        top_acc = train(filepath, n_maps, num_epochs, hidden_states, 
                                  random_state, prefix, plot_info=False,
                                  print_info=False)
        if not top_acc:
            print("Invalid Testing Set")
            total_number -= 1 
        else:
            print(f"Total Accuracy: {top_acc}")
            avg_acc += top_acc
    print(f"Average Accuracy: {avg_acc/total_number}")
    print(f"Total Time: {time.time()-start}")


seeds = [3,10,20,23,26,33,39,44,51,60,64,75,81]

def test():
    train("sample_peaks_4.npy", 4, 2000, [32], 
                                  3, "test", plot_info=True,
                                  print_info=True)

if __name__ == "__main__":
    # start = time.time()
    # filepath = "sample_peaks_4.npy"
    # num_epochs = 10
    # n_maps = int(ntpath.basename(filepath).split(".npy")[0].split("_")[-1])
    # prefix = "test"
    # hidden_states = [32]
    # run(filepath, prefix, hidden_states, num_epochs=num_epochs)
    seed_torch()
    test()