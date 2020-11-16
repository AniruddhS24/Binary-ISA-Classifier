import src.Server as svr
import json
import binascii
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.utils import *

'''
Using the server, we can build our dataset by simply querying for random binary blobs to use as 
training examples. Since we can access the target architecture, we can easily label our dataset.
Moreover, to keep our data balanced with respect to the 12 classes, we generate num_samples examples
from each class, so our total dataset size will be num_samples*12.
'''
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
            seen_classes[s.ans]+=1 #we have an example from this class
            if(seen_classes[s.ans]==num_samples):
                num_done+=1 #generated enough examples for this class
            data.append((str(bin_hex), s.ans)) #add training example
        if num_done == 12:
            break #break when all 12 classes are done

    print("Extracted " + str(len(data)) + " training examples from server")
    with open(filename, 'w+') as f:
        json.dump(data, f)

'''
Read data from datafile and format our training data properly. For this task,
we can adapt a simple bag-of-words representation which is simply an array of
counts/frequencies for each word. Also takes a gram parameter which denotes 
the length of individual 'tokens' or words we split our binary blob into
'''
def read_data_rawcounts(filename, gram):
    with open(filename) as f:
        data_text = json.load(f)

    #utility class to index vocabulary and labels, see utils.py
    vocab = Indexer()
    labels = Indexer()
    for datapoint in data_text:
        for ci in range(0, len(datapoint[0])-gram+1):
            vocab.add_and_get_index(str(datapoint[0][ci:ci+gram])) #add word to vocabulary
            labels.add_and_get_index(datapoint[1]) #add to labels if not present

    #define numpy arrays for training data
    train_x = np.zeros(shape=(len(data_text), len(vocab)))
    train_y = np.zeros(shape=(len(data_text), len(labels)))

    for i in range(len(data_text)):
        for ci in range(0, len(data_text[i][0])-gram+1):
            train_x[i][vocab.index_of(data_text[i][0][ci:ci+gram])] += 1 #bag-of-words counts
    for i in range(len(data_text)):
        train_y[i][labels.index_of(data_text[i][1])] = 1 #one-hot encoding

    return vocab, labels, train_x, train_y

'''
Reads data as raw counts but also applies tf-idf weighting. Applying tf-idf can
be advantageous as it normalizes our raw counts and also assigns smaller weight 
to common terms, thereby giving rare/special terms more weight. I calculated it
as follows:

tf(t, d) = rawcount(t, d)/sum(rawcount(d))
idf(t) = log(N / df(t))

tfidf(t, d) = tf(t,d) * idf(t)
    
    for term t in document d, and where our corpus has N documents. df(t) represents
    the number of documents which contain at least 1 occurrence of term t. 

For example, in
an NLP setting, a word like "the" is probably not very useful for a text 
classification task because its so common. If we simply use raw counts, our model
will see a large weight assigned to "the" and might treat it as an important word. 
However, the idf term for "the" is probably very low as it most likely appears in almost
all documents, so the tf-idf weight for "the" will be low, as desired. Similarly, we want
to assign higher weight to more rare tokens in our binary blob that can help predict the target
architecture
'''
def read_data_tfidf(filename, gram):
    vocab, labels, train_x, train_y = read_data_rawcounts(filename, gram)
    for i in range(train_x.shape[0]): #tf
        train_x[i] /= np.sum(train_x[i])
    for w in range(train_x.shape[1]): #idf
        train_x[:,w] *= np.log(train_x.shape[1]/np.sum((train_x[:,w] > 0)))
    return vocab, labels, train_x, train_y

'''
This function is simply used to visualize our data for a set of labels. Scales data and then
applies PCA to 1 or 2 dimensions, and plots datapoints. PCA is a method by which 
'''
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

