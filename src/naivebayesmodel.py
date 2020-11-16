import numpy as np
from src.dataprocessing import *

'''
NB Classifier
    A naive bayes classifier, which is a probabilistic classifier which uses Bayes theorem to predict
    class probabilities. Receives binary blob count vector as input and calculates P(class | input blob)
'''
class NaiveBayesClassifier:
    def __init__(self, gram, tf_idf, vocab:Indexer, labels:Indexer, probs, train_x, train_y):
        self.gram = gram
        self.tf_idf = tf_idf
        self.vocab = vocab
        self.labels = labels
        self.probs = probs
        self.train_x = train_x
        self.train_y = train_y

    '''
    Convert input binary blob string to BOW representation, applying tf-idf if necessary
    '''
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

    '''
    Predict class as follows:
    
    P(class | input blob) = P(class) * P(input blob | class) / P(input blob)
        where we select the class which yields the maximum posterior probability P(class | input blob)
    
    We can calculate P(input blob | class) from our training data - assuming blob words/tokens are independent, 
    we have P(input blob | class) = P(t_1 | class) * P(t_2 | class) * ... * P(t_n | class) where each t_i is a token
    of length gram. We can calculate any P(t_i | class) as follows:
    
        P(t_i | class) = occurrences of t_i in examples labeled class/ total occurrences of t_i
    '''
    def predict(self, x, targets):
        best_prob = -1000000007
        best_class = ""
        x = self.process_input(x)
        for pred_class in targets:
            cur_prob = 0.0
            for w in range(len(x)):
                cur_prob += x[w]*self.probs[self.labels.index_of(pred_class), w]
            # for c in range(0, len(x)-self.gram+1):
            #     if(self.vocab.contains(x[c:c+self.gram])):
            #         tokenid = self.vocab.index_of(x[c:c+self.gram])
            #         classid = self.labels.index_of(pred_class)
            #         cur_prob += self.probs[classid, tokenid]
            if(cur_prob > best_prob):
                best_prob = cur_prob
                best_class = pred_class

        return best_class

'''
Not really a training procedure, but computes probability matrix P(t_i | class) for all 12 classes
Note that
    P(t_i | class) = occurrences of t_i in examples labeled class/ total occurrences of t_i
'''
def train_naive_bayes(datafilename, gram, tf_idf = False):
    # get data
    if tf_idf:
        vocab, labels, train_x, train_y = read_data_tfidf(datafilename, gram=gram)
    else:
        vocab, labels, train_x, train_y = read_data_rawcounts(datafilename, gram=gram)

    # label_based is a numpy array where label_based[t_i][class] = P(t_i | class)
    label_based = np.zeros(shape=(len(labels), train_x.shape[1]))

    for label in labels.objs_to_ints.keys():
        id = labels.index_of(label)
        # select all training examples of label
        classdata = train_x[(train_y[: , id]==1).nonzero()]
        # take the sum of counts across all training examples to obtain # occurrences of t_i in examples of label
        sum_labels = np.sum(classdata,axis=0)

        # laplace smoothing with alpha = 1, this is needed to ensure probabilities of 0 do not arise
        sum_labels += 1
        label_based[id] = np.divide(sum_labels,np.sum(sum_labels, axis=0) + len(vocab))

    # log probabilities are more convenient
    label_based = np.log(label_based)

    nb = NaiveBayesClassifier(gram, tf_idf, vocab, labels, label_based, train_x, train_y)
    return nb
