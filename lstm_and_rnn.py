# 
# Author: Normand Overney 
#

import torch
import torch.nn as nn
from test_rnn import RNNModel
from test_lstm import LSTMModel
from test_lstm import create_data

import time
import numpy as np

# the block size is 100 so I should have seperate blocks to get seperate classifications 
# across the whole range of the sequence
def create_block(x, chunk_size):
    # this will form one block of multiple chunks
    block = [x[i:i+chunk_size] for i in range(len(x)-chunk_size)]
    return torch.Tensor(block)

class CombinedModels(nn.Module):
    def __init__(self, rnn, lstm, input_dim_rnn, input_dim_lstm, hidden_dim, output_dim):
        super(CombinedModels, self).__init__()
        self.rnn = rnn
        self.lstm = lstm
        self.chunk_size = input_dim_rnn
        self.input_dim_lstm = input_dim_lstm
        self.input_dim_rnn = input_dim_rnn

        input_size = (input_dim_lstm-input_dim_rnn+1)*2
        hidden_sizes = [hidden_dim, 100] # pass in an array later on 
        output_size = output_dim
        self.combined = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                      nn.ReLU(),
                                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                      nn.ReLU(),
                                      nn.Linear(hidden_sizes[1], output_size),
                                      nn.Softmax(dim=1))
        
    def forward(self, x):
        seq = torch.Tensor(x)
        seq = seq.view(1,-1,self.input_dim_lstm).requires_grad_()
        head_lstm = self.lstm(seq)
        # covert x into a bunch of batches of a certain sequence length
        block = create_block(x, self.chunk_size)
        block = block.view(block.size()[0], -1, self.input_dim_rnn).requires_grad_()
        head_rnn = self.rnn(block)
        # want to flatten the head rnn to have the class predictions side by side
        head_rnn = head_rnn.view(head_rnn.size()[1]*head_rnn.size()[0],1)
        #print(head_rnn.size())
        #print(head_lstm.size())
        head_lstm = head_lstm.view(head_lstm.size()[1], head_lstm.size()[0])
        x = torch.cat((head_lstm, head_rnn), dim=0)
        x = x.view(x.size()[1], x.size()[0])
        return self.combined(x)


# we need our own prediction 
def predict_by_sample(model, test_features, test_labels,input_dim_lstm):
    # we can just take the majority vote 
    for images, labels in zip(test_features, test_labels):
        for i in range(len(images)-input_dim_lstm):
            model.train()
            image = images[i:i+input_dim_lstm]

            # Forward pass to get output/logits
            outputs = model(image)

            label = np.array([labels]) 
            label = torch.LongTensor(label)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # just realized that I just need to take the fifty percent majority 
            flags = (predicted == label).data.tolist()
            assert len(flags) == 1

            print(f"labels:{labels}, flags:{flags}")
    

def train(filepath, input_dim_rnn, input_dim_lstm, layer_dim_rnn, layer_dim_lstm, 
          hidden_dim_rnn, hidden_dim_lstm):
    # we have the model rnn model
    # I have to get the two trained models 
    train_features, test_features, train_labels, test_labels = create_data(filepath)


    output_dim = 2
    rnn = RNNModel(input_dim_rnn, hidden_dim_rnn, layer_dim_rnn, output_dim)
    lstm = LSTMModel(input_dim_lstm, hidden_dim_lstm, layer_dim_lstm, output_dim)
    
    
    # we have to figure out the input dim for each sample so once we zero fill them 
    hidden_dim_combined = 250
    model = CombinedModels(rnn, lstm, input_dim_rnn, input_dim_lstm, hidden_dim_combined, output_dim)

    # go through the samples like the train for lstm

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.001

    num_epochs = 100
    plot_every = 100

    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=weight_decay)

    # so wer are not using this schedular since it does not work 
    # lmbda = lambda epoch: 0.95
    #scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    # Assuming optimizer uses lr = 0.05 for all groups
    # lr = 0.05     if epoch < 30
    # lr = 0.005    if 30 <= epoch < 80
    # lr = 0.0005   if epoch >= 80
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,75], gamma=0.1)

    iter = 0
    for epoch in range(num_epochs):
        epoch_start = time.time()
        for i, (images, labels) in enumerate(zip(train_features, train_labels)):#train_loader):

            # we need to split the images based on the size of input_dim for lstm
            for j in range(len(images)-input_dim_lstm):
                model.train()
                # Load images as tensors with gradient accumulation abilities
                # dont need to change the shape since it is already at ideal shape 
                # torch.Size([100, 10])
                #print(images.shape)
                image = images[j:j+input_dim_lstm]

                #print(images.size())
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()

                # Forward pass to get output/logits
                # outputs.size() --> 100, 10
                outputs = model(image)

                # Calculate Loss: softmax --> cross entropy loss
                # I need to make the labels one large array 
                label = np.array([labels]) # there are two labels since 
                # it is the combined result 
                label = torch.LongTensor(label)
                #print(outputs)
                #print(label)
                loss = criterion(outputs, label)

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                optimizer.step()

                iter += 1
                # Add current loss avg to list of losses
                if iter % plot_every == 0:
                    model.eval()
                    results = predict_by_sample(model, test_features, test_labels,input_dim_lstm)
                    # Print Loss
                    print('Iteration: {}. Loss: {}'.format(iter, loss.item()))
        scheduler.step()
        print(f"finished epoch: {epoch}")


if __name__ == "__main__":
    filepath = "data/sample_peaks_4.npy"
    chunk_size = 25
    block_size = 100

    num_layers_rnn = 3
    num_layers_lstm = 2

    num_hidden_rnn = 1500
    num_hidden_lstm = 1250

    train(filepath, chunk_size, chunk_size*block_size, num_layers_rnn, num_layers_lstm, 
          num_hidden_rnn, num_hidden_lstm)