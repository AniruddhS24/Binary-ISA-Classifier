import src.Server as svr
import json
import binascii
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.utils import *

def extract_data(filename, num_samples):
    s = svr.Server()
    data = []
    seen_classes = dict()
    num_done = 0
    while True:
        s.get()
        bin_hex = binascii.hexlify(s.binary)
        s.post(s.targets[0])
        if s.ans not in seen_classes.keys():
            seen_classes[s.ans] = 0
        if seen_classes[s.ans] < num_samples:
            seen_classes[s.ans]+=1
            if(seen_classes[s.ans]==num_samples):
                num_done+=1
            data.append((str(bin_hex), s.ans))
        if num_done == 12:
            break

    print(len(data))
    with open(filename, 'w+') as f:
        json.dump(data, f)

def read_data_tfidf(filename, gram):
    vocab, labels, train_x, train_y = read_data_rawcounts(filename, gram)
    for i in range(train_x.shape[0]): #tf
        train_x[i] /= np.sum(train_x[i])
    for w in range(train_x.shape[1]): #idf
        train_x[:,w] *= np.log(train_x.shape[1]/np.sum((train_x[:,w] > 0)))
    return vocab, labels, train_x, train_y

def read_data_rawcounts(filename, gram):
    with open(filename) as f:
        data_text = json.load(f)

    vocab = Indexer()
    labels = Indexer()
    for datapoint in data_text:
        for ci in range(0, len(datapoint[0])-gram+1):
            vocab.add_and_get_index(str(datapoint[0][ci:ci+gram]))
            labels.add_and_get_index(datapoint[1])

    train_x = np.zeros(shape=(len(data_text), len(vocab)))
    train_y = np.zeros(shape=(len(data_text), len(labels)))

    for i in range(len(data_text)):
        for ci in range(0, len(data_text[i][0])-gram+1):
            train_x[i][vocab.index_of(data_text[i][0][ci:ci+gram])] += 1
    for i in range(len(data_text)):
        train_y[i][labels.index_of(data_text[i][1])] = 1

    return vocab, labels, train_x, train_y

def visualize_data(datafile, gram, view_labels, dim):
    vocab, labels, train_x, train_y = read_data_tfidf(datafile, gram)
    train_x = StandardScaler().fit_transform(train_x)
    if dim==2:
        pca = PCA(n_components=2)
        comps = pca.fit_transform(train_x)

        for lbl in view_labels:
            lidx = labels.index_of(lbl)
            plotdata = comps[np.where(train_y[:, lidx]==1)]
            plt.scatter(plotdata[:,0], plotdata[:,1], label = lbl)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=4)
        plt.show()
    elif dim==3:
        pca = PCA(n_components=3)
        comps = pca.fit_transform(train_x)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for lbl in view_labels:
            lidx = labels.index_of(lbl)
            plotdata = comps[np.where(train_y[:, lidx] == 1)]
            ax.scatter(plotdata[:, 0], plotdata[:, 1], plotdata[:,2], label=lbl)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=4)
        plt.show()


if __name__ == '__main__':
    visualize_data("../data/datafile150.json", 3, ['sparc', 'xtensa', 'powerpc', 'sh4', 'arm'], dim=3)

