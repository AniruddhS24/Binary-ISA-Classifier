import numpy as np
from src.dataprocessing import *

class NaiveBayesClassifier:
    def __init__(self, gram, tf_idf, vocab:Indexer, labels:Indexer, probs, train_x, train_y):
        self.gram = gram
        self.tf_idf = tf_idf
        self.vocab = vocab
        self.labels = labels
        self.probs = probs
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
                x_arr[w] *= np.log(self.train_x.shape[1] / np.sum((self.train_x[:, w] > 0)))
        return x_arr

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



def train_naive_bayes(datafilename, gram, tf_idf = False):
    if tf_idf:
        vocab, labels, train_x, train_y = read_data_tfidf(datafilename, gram=gram)
    else:
        vocab, labels, train_x, train_y = read_data_rawcounts(datafilename, gram=gram)

    label_based = np.zeros(shape=(len(labels), train_x.shape[1])) #probability of x_i in each class, P(x_i | y)
    for label in labels.objs_to_ints.keys():
        id = labels.index_of(label)
        classvalues = train_x[(train_y[: , id]==1).nonzero()]
        sum_labels = np.sum(classvalues,axis=0)
        sum_labels += 1
        label_based[id] = np.divide(sum_labels,np.sum(sum_labels, axis=0) + len(vocab)) #compute probs w/ smoothing
    label_based = np.log(label_based)
    nb = NaiveBayesClassifier(gram, tf_idf, vocab, labels, label_based, train_x, train_y)
    return nb

if __name__ == '__main__':
    model = train_naive_bayes("../data/datafile100.json")
    #vocab, labels, train_x, train_y = read_data_rawcounts("../data/datafile500.json")
    print(model.predict("b'00017a5200017c0e011b0c0f600000002000000018000000000000012400448d0b8e0a8f09460ec00102decfcecd0e600065000001000000000000000044dc86'"))
    print(model.predict("b'000000008345fc018b45fc3b45e47ccf837de0007425488b1500000000b8ffffffff4831c2488b45d84889c6bf00000000b800000000e80000000090c9c35548'"))
    print(model.predict("b'0083a0000065280000e1b0000081b00000e1b00200e373000098280300e473000061b0000041b01f04ff470000652c0300972cc2047748430d97480304434402'"))
    print(model.predict("b'00622803f07f440300e343000061b0000062200000a22c0300832cc504a348430d83480304a344000061b0000042200500a22c0800822c05006220c504a34843'"))
    print(model.predict("b'0000909100001a8219822c2f3d2f2f5f3f4fe0910000f091000080819181877b9f7991838083e0910000f09100003183208389819a8101969a83898318161906'"))
