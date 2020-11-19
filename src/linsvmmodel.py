import numpy as np
from src.dataprocessing import *

'''
Class which encapsulates multiple SVMs to make a prediction. 
Note: this class uses one vs. one prediction across the SVMs - I couldn't figure out a way
to use one vs. all here because SVMs don't produce probabilities. However, despite having
12c2 = 66 SVMs this approach seemed to work alright, though I'm sure better alternatives exist.
'''
class SVMMultiClass:
    def __init__(self, grams, tf_idf, pairs, svms, vocab: Indexer, labels: Indexer, train_x, train_y):
        self.grams = grams
        self.tf_idf = tf_idf
        self.pairs = pairs #list of 12c2 class pairs (i.e. [mipsel, sh4] ) which correspond to each SVM
        self.svms = svms #list of SVM models
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
        votes = np.zeros(len(self.labels))
        for i in range(len(self.svms)):
            pd = self.svms[i].predict(x)
            votes[self.pairs[i][pd]]+=1 # increment votes for this prediction

        maxvotes = 0
        res = targets[0]
        for target in targets:
            if votes[self.labels.index_of(target)] > maxvotes:
                maxvotes = votes[self.labels.index_of(target)]
                res = target # choose target with maximum votes

        return res

'''
SVM
    Linear SVM which is trained for a binary classification task between two binary blob architectures.
'''
class SVM:
    def __init__(self, wlen):
        self.w = np.zeros(wlen)  # weights array
        self.b = 0  # bias term

    '''
    Routine to update weights while training. L2 regularization is used.
    '''
    def update(self, x, y, reg_c, lr):
        z = np.dot(self.w,x) - self.b

        # hinge loss is 0, gradient only involves regularization term
        if y*z >= 1:
            grads = self.w
            self.w -= lr*grads #update weights
        # hinge loss nonzero
        else:
            grads = -reg_c*y*x + self.w
            self.w -= lr*grads #update weights
            self.b -= lr*y

    '''
    Return 1/0 prediction for SVM
    '''
    def predict(self, x):
        z = np.dot(self.w,x) - self.b
        if z >= 0:
            return 1
        else:
            return 0

'''
Helper function for min-max scaling of training data (use if needed)
'''
def scale_data(train_x):
    mins = np.min(train_x, axis=0)
    maxs = np.max(train_x, axis=0)
    train_x = (train_x - mins)/(maxs-mins)
    return train_x

'''
Train 12c2 SVMs and return an SVMMulticlass model.
'''
def train_SVM(datafilename, grams, tf_idf, EPOCHS = 8, lr = 0.001, decay = 0.0001, reg_c = 1000):
    # get data
    vocab, labels, train_x_raw, train_y = read_data_rawcounts(datafilename, grams=grams)
    train_x = np.copy(train_x_raw)
    if tf_idf:
        apply_tfidf(train_x, train_x_raw)


    # generate 12c2 pairs of classes, and initialize array of untrained SVMs
    x = [i for i in range(len(labels))]
    pairs = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
    svms = [SVM(len(vocab)) for _ in range(len(pairs))]


    # training
    for epoch in range(EPOCHS):
        perm = np.random.permutation(train_x.shape[0]) #shuffle examples
        for i in range(len(perm)):
            for svmidx in range(len(svms)):
                classname = np.where(train_y[perm[i]]==1)[0][0]
                if pairs[svmidx][0] == classname:
                    y = -1
                elif pairs[svmidx][1] == classname:
                    y = 1
                else:
                    continue
                svms[svmidx].update(train_x[perm[i]],y, reg_c, lr)
        lr = lr * (1.0 / (1.0 + decay*epoch)) #decay learning rate

    return SVMMultiClass(grams, tf_idf, pairs, svms, vocab, labels, train_x_raw, train_y)

# hyperparameter tuning
if __name__ == '__main__':
    params = dict()
    params['epochs'] = [5, 7, 12]
    params['lr'] = [0.0001, 0.001, 0.01]
    params['reg_c'] = [0.5, 10, 500, 1000]

    with open('../data/datafile20.json') as f:
        dev_set = json.load(f)


    for epoch in params['epochs']:
        for lr in params['lr']:
            for reg_c in params['reg_c']:
                avg_acc = 0.0
                for i in range(3):
                    model = train_SVM('../data/datafile150.json', grams=[2,3], tf_idf=True,
                                     EPOCHS = epoch, lr = lr, reg_c = reg_c)
                    num_correct = 0
                    for datapoint in dev_set:
                        if model.predict(datapoint[0], list(model.labels.objs_to_ints.keys())) == datapoint[1]:
                            num_correct += 1
                    avg_acc += num_correct/len(dev_set)
                print("epochs %d    lr %f     reg_c %f       accuracy %f" % (epoch, lr, reg_c, avg_acc/3))

#lr = 0.001 and regc = 1000 works well, train for 7 epochs