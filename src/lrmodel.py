import numpy as np
from src.dataprocessing import *

'''
Class which encapsulates multiple LR models to make a one vs. all prediction. 
The class which returns the maximum probability
as returned by the LR model is the predicted class.
'''
class LRClassifier:
    def __init__(self, grams, tf_idf, lrs, vocab: Indexer, labels: Indexer, train_x, train_y):
        self.grams = grams
        self.tf_idf = tf_idf
        self.lrs = lrs  # list of SVM models
        self.vocab = vocab
        self.labels = labels
        self.train_x = train_x
        self.train_y = train_y

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
    Iterate through SVMs while counting the number of predictions made for some class as
    'votes' - the class with most votes is likely the target class.
    '''
    def predict(self, x, targets):
        x = self.process_input(x)
        maxprob = 0.0
        res = None
        for target in targets:
            id = self.labels.index_of(target)
            prob = self.lrs[id].predict(x)
            if prob > maxprob:
                maxprob = prob
                res = target

        return res

'''
LR
    LR model that does binary classification between some ISA and all other ISAs. 
'''
class LR:
    def __init__(self, wlen):
        self.w = np.zeros(wlen)
        self.b = 0

    def activ(self, x):
        if x < 0:
            return 1 - 1 / (1 + np.exp(x))
        else:
            return 1 / (1 + np.exp(-x))
    '''
        Routine to update weights while training. L2 regularization is used.
    '''
    def update(self, x, y, reg_lambda, lr):
        z = self.activ(np.dot(self.w, x) + self.b)
        self.w  -= lr*(z-y)*x + lr*2*reg_lambda*self.w
        self.b -= lr*(z-y)

    def predict(self, x):
        z = self.activ(np.dot(self.w, x) + self.b)
        return z

'''
Helper function for min-max scaling of training data (use if needed)
'''
def scale_data(train_x):
    mins = np.min(train_x, axis=0)
    maxs = np.max(train_x, axis=0)
    train_x = (train_x - mins)/(maxs-mins)
    return train_x

'''
Train 12 LR classifiers, each of which is trained to predict one of the 12 ISA architectures vs. other architectures.
'''
def train_LR(datafilename, grams, tf_idf):
    # get data
    vocab, labels, train_x_raw, train_y = read_data_rawcounts(datafilename, grams=grams)
    train_x = np.copy(train_x_raw)
    if tf_idf:
        apply_tfidf(train_x, train_x_raw)

    lrs = [LR(len(vocab)) for _ in range(len(labels))]

    # hyperparameters
    EPOCHS = 10
    lr = 0.1
    decay = 0.0001
    reg_lambda = 0

    # training
    for epoch in range(EPOCHS):
        perm = np.random.permutation(train_x.shape[0])  # shuffle examples
        for i in range(len(perm)):
            for lrid in range(len(lrs)):
                classid = np.where(train_y[perm[i]]==1)[0][0]
                if lrid == classid: # this is the LR model which should predict positive on this ISA class
                    lrs[lrid].update(train_x[perm[i]], 1, reg_lambda, lr)
                else:
                    lrs[lrid].update(train_x[perm[i]], 0, reg_lambda, lr)
        lr = lr * (1.0 / (1.0 + decay * epoch))  # decay learning rate

    return LRClassifier(grams, tf_idf, lrs, vocab, labels, train_x_raw, train_y)