import numpy as np
import torch
import torch.nn as nn
from src.dataprocessing import *

'''
Multilayer perceptron
    2-layer neural network to predict class label given binary blob count vector as input
'''
class MLP(nn.Module):
    def __init__(self, inp_shape, hid_shape, op_shape):
        super(MLP, self).__init__()

        self.dense1 = nn.Linear(inp_shape, hid_shape)
        self.act1 = nn.LeakyReLU()
        self.dense2 = nn.Linear(hid_shape, hid_shape)
        self.act2 = nn.LeakyReLU()
        self.dpout2 = nn.Dropout(p=0.2)
        self.dense3 = nn.Linear(hid_shape, op_shape)
        self.softmx = nn.LogSoftmax(dim=1)

        nn.init.kaiming_uniform_(self.dense1.weight)
        nn.init.kaiming_uniform_(self.dense2.weight)

    def forward(self, x):
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        x = self.dpout2(x)
        x = self.dense3(x)
        x = self.softmx(x)
        return x

'''
Classifier which outputs prediction from trained neural net
'''
class MLPClassifier:
    def __init__(self, grams, tf_idf, vocab: Indexer, labels: Indexer, train_x, train_y, net:MLP):
        self.grams = grams
        self.tf_idf = tf_idf
        self.vocab = vocab
        self.labels = labels
        self.train_x = train_x
        self.train_y = train_y

        self.net = net #trained MLP

    '''
    Convert input binary blob string to BOW representation, applying tf-idf if necessary
    '''
    def process_input(self, x):
        x_arr = np.zeros(len(self.vocab))
        for gram in self.grams:
            for i in range(0, len(x) - gram + 1):
                if (self.vocab.contains(x[i:i + gram])):
                    x_arr[self.vocab.index_of(x[i:i + gram])] += 1
        if self.tf_idf:
            apply_tfidf(x_arr, self.train_x)
        return x_arr

    '''
    Predict class as argmax of output probabilities from net
    '''
    def predict(self, x, targets):
        self.net.eval()
        x = self.process_input(x)
        x = x.astype(np.float32)
        pred = self.net.forward(torch.from_numpy(x).unsqueeze(dim=0)).squeeze()
        max_prob = pred[0]
        res = targets[0]
        for target in targets:
            if pred[self.labels.index_of(target)] > max_prob:
                max_prob = pred[self.labels.index_of(target)]
                res = target
        return res

'''
Helper function for min-max scaling of training data (use if needed)
'''
def scale_data(train_x): #minmax scaling probably not good
    mins = np.min(train_x, axis=0)
    maxs = np.max(train_x, axis=0)
    train_x = (train_x - mins)/(maxs-mins)
    return train_x

'''
Train MLP and return a MLPClassifier model
'''
def train_MLP(datafilename, grams, tf_idf):
    # get data
    vocab, labels, train_x_raw, train_y = read_data_rawcounts(datafilename, grams=grams)
    train_x = np.copy(train_x_raw)
    if tf_idf:
        apply_tfidf(train_x, train_x_raw)

    # scale data and convert numpy arrays to type float32 (needed for training)
    train_x = scale_data(train_x)
    train_x = train_x.astype(np.float32)
    train_y = train_y.astype(np.float32)

    # define untrained model
    model = MLP(train_x.shape[1], 64, train_y.shape[1])

    # hyperparameters
    EPOCHS = 20
    batch_size = 10
    lr = 0.002

    # our optimizer is Adam
    optimizer = torch.optim.Adam(params=model.parameters(), lr = lr)

    # choose loss function as negative log-likelihood loss
    criterion = nn.NLLLoss()

    # training
    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(train_x.shape[0]) #shuffle examples
        total_loss = 0.0
        for i in range(0, train_x.shape[0], batch_size):
            optimizer.zero_grad()
            # get prediction
            pred = model.forward(torch.from_numpy(train_x[perm[i:i+batch_size], :]))
            # compute loss
            loss = criterion(pred, torch.argmax(torch.from_numpy(train_y[perm[i:i+batch_size], :]), dim=1))
            total_loss += loss*batch_size
            # backpropagate to compute gradients
            loss.backward()
            # update model weights
            optimizer.step()
        print("Epoch: %d    Loss: %f" % (epoch, total_loss.item()))

    mlpc = MLPClassifier(grams, tf_idf, vocab, labels, train_x_raw, train_y, net=model)
    return mlpc
