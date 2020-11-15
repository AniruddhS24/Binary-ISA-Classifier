import numpy as np
import torch
import torch.nn as nn
from src.dataprocessing import *

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

class MLPClassifier:
    def __init__(self, gram, tf_idf, vocab: Indexer, labels: Indexer, train_x, train_y, net:MLP):
        self.gram = gram
        self.tf_idf = tf_idf
        self.vocab = vocab
        self.labels = labels
        self.train_x = train_x
        self.train_y = train_y

        self.net = net

    def process_input(self, x):
        x_arr = np.zeros(len(self.vocab))
        for i in range(0, len(x) - self.gram + 1):
            if (self.vocab.contains(x[i:i + self.gram])):
                x_arr[self.vocab.index_of(x[i:i + self.gram])] += 1
        if self.tf_idf:
            x_arr /= np.sum(x_arr)
            for w in range(len(x_arr)):
                x_arr[w] *= np.log(self.train_x.shape[1] / np.sum((self.train_x[:, w] > 0)))
        return x_arr

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

def scale_data(train_x): #minmax scaling probably not good
    mins = np.min(train_x, axis=0)
    maxs = np.max(train_x, axis=0)
    train_x = (train_x - mins)/(maxs-mins)
    return train_x

def train_MLP(datafilename, gram, tf_idf):
    if tf_idf:
        vocab, labels, train_x, train_y = read_data_tfidf(datafilename, gram=gram)
    else:
        vocab, labels, train_x, train_y = read_data_rawcounts(datafilename, gram=gram)

    train_x = scale_data(train_x)
    train_x = train_x.astype(np.float32)
    train_y = train_y.astype(np.float32)
    model = MLP(train_x.shape[1], 64, train_y.shape[1])
    EPOCHS = 20
    batch_size = 10
    lr = 0.001

    optimizer = torch.optim.Adam(params=model.parameters(), lr = lr)
    criterion = nn.NLLLoss()
    model.train() #set to training mode
    for epoch in range(EPOCHS):
        perm = torch.randperm(train_x.shape[0])
        total_loss = 0.0
        for i in range(0, train_x.shape[0], batch_size):
            optimizer.zero_grad()
            pred = model.forward(torch.from_numpy(train_x[perm[i:i+batch_size], :]))
            loss = criterion(pred, torch.argmax(torch.from_numpy(train_y[perm[i:i+batch_size], :]), dim=1))
            total_loss += loss*batch_size
            loss.backward()
            optimizer.step()
        print("Epoch: %d    Loss: %f" % (epoch, total_loss.item()))

    mlpc = MLPClassifier(gram, tf_idf, vocab, labels, train_x, train_y, net=model)
    return mlpc
