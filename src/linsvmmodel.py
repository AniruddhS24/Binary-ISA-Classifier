import numpy as np
from src.dataprocessing import *
#hello
class SVMMultiClass:
    def __init__(self, gram, tf_idf, pairs, svms, vocab: Indexer, labels: Indexer, train_x, train_y):
        self.gram = gram
        self.tf_idf = tf_idf
        self.pairs = pairs
        self.svms = svms
        self.vocab = vocab
        self.labels = labels
        self.train_x = train_x
        self.train_y = train_y

    def process_input(self, x):
        x_arr = np.zeros(len(self.vocab))
        for i in range(0, len(x) - self.gram + 1):
            if (self.vocab.contains(x[i:i + self.gram])):
                x_arr[self.vocab.index_of(x[i:i + self.gram])] += 1
        if self.tf_idf:
            x_arr /= np.sum(x_arr)
            for w in range(len(x_arr)):
                if np.sum((self.train_x[:, w] > 0)) > 0:
                    x_arr[w] *= np.log(self.train_x.shape[1] / np.sum((self.train_x[:, w] > 0)))
        return x_arr

    def predict(self, x, targets):
        x = self.process_input(x)
        votes = np.zeros(len(self.labels))
        for i in range(len(self.svms)):
            pd = self.svms[i].predict(x)
            votes[self.pairs[i][pd]]+=1

        maxvotes = 0
        res = targets[0]
        for target in targets:
            if votes[self.labels.index_of(target)] > maxvotes:
                maxvotes = votes[self.labels.index_of(target)]
                res = target

        return res

class SVM:
    def __init__(self, wlen):
        self.w = np.zeros(wlen)
        self.b = 0

    def update(self, x, y, reg_c, reg_lambda, lr):
        z = np.dot(self.w,x) - self.b

        if y*z >= reg_c:
            grads = 2*reg_lambda*self.w
            self.w -= lr*grads
        else:
            grads = -y*x + 2*reg_lambda*self.w
            self.w -= lr*grads
            self.b -= lr*y

    def predict(self, x):
        z = np.dot(self.w,x) - self.b
        if z >= 0:
            return 1
        else:
            return 0

def scale_data(train_x): #minmax scaling probably not good
    mins = np.min(train_x, axis=0)
    maxs = np.max(train_x, axis=0)
    train_x = (train_x - mins)/(maxs-mins)
    return train_x

def train_SVM(datafilename, gram, tf_idf):
    if tf_idf:
        vocab, labels, train_x, train_y = read_data_tfidf(datafilename, gram=gram)
    else:
        vocab, labels, train_x, train_y = read_data_rawcounts(datafilename, gram=gram)
    x = [i for i in range(len(labels))]
    pairs = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
    svms = [SVM(len(vocab)) for _ in range(len(pairs))]

    EPOCHS = 7
    #0.005 1 0.5
    #0.01 2 0.01

    lr = 0.02
    reg_c = 3
    reg_lambda = 0.01
    for epoch in range(EPOCHS):
        perm = np.random.permutation(train_x.shape[0])
        for i in perm:
            for svmidx in range(len(svms)):
                classname = np.where(train_y[i]==1)[0][0]
                if pairs[svmidx][0] == classname:
                    y = -1
                elif pairs[svmidx][1] == classname:
                    y = 1
                else:
                    continue
                svms[svmidx].update(train_x[i],y, reg_c, reg_lambda, lr)
    svmmc = SVMMultiClass(gram, tf_idf, pairs, svms, vocab, labels, train_x, train_y)
    return svmmc
