# Binary ISA Classifier

This is my attempt for the Praetorian Machine Learning Binaries (MLB) challenge. Here,
I will briefly describe my approach and choice of models.

You can run the classifier and select a model in ```runprgm.py```. The models and
data processing routines are in ```src```, and the training data files are in ```data```.

## Understanding the Task

The task at hand is: given a random binary blob, predict its instruction set architecture
(ISA). Essentially, we need to find a function that maps binary blobs to one of the 12
ISAs provided. Framing this as a supervised learning task, we can let our inputs be
some representation of a binary blob while the output is a class label for the ISA.

## Data Processing

All data processing routines can be found in ```src/dataprocessing.py``` where I generate
datasets, process input, and visualize data. 

Using the server, we can build our dataset by simply querying for random binary blobs to use as 
training examples. These files are stored in ```data\datafile<num_samples>.json``` where
```num_samples``` denotes the number of training examples for each of the 12 classes.
 
Since we also have access to the target architecture, we now have a fully labeled dataset.
Now, each training example is a (binary blob, target architecture) pair.
Moreover, to keep our data balanced with respect to the 12 classes, we generate an equal number of 
training examples from each class.

Upon some examination, I found that this is essentially a text classification problem -
first, we can convert our binary blobs to hexadecimal, and we can tokenize each blob by
chunks of hex characters. This was also suggested in the MLB article, so it seemed like a good
starting point. Viewing this challenge from an NLP setting now, we can create features
by extracting n-grams as sequences of hex characters (i.e. 'c30' or '17e'). The vocabulary will then be all of the
words (hex chunks) present in the training data. Upon experimenting, I found that bigrams and trigrams
worked well for this task.

Moreover, for each binary blob training example, we can simply represent it as an array of 'word' frequencies.
Our array size will be equal to the length of the vocabulary, and the relative order of words is not preserved - 
this is the standard bag-of-words (BOW) representation we will use. A BOW model would be suboptimal for many real-world NLP tasks - however, for this 
type of text classification task, I found that it's sufficient. 

However, storing raw counts poses an issue - upon examining the distribution of counts, some hex chunks such as
'00', '000', or '10' were extremely common across all ISA classes. These features are then probably not very useful,
yet they are assigned a high raw count value, so the model perceives them as important. To fix this, a common NLP 
technique is to use tf-idf weighting. 

While the term frequency may be high for '00', the idf score will be low, causing its tf-idf weight to be low.
By applying this weight, stop words/common hex chunks will receive low tf-idf scores, and hex chunks which are
more representative of the target architecture will ideally receive higher tf-idf scores.

However, even after this, we are likely to end up with very sparse BOW vectors as the vocabulary size
is very large (~4500 words) in contrast with each training example.

Although there are more ways to further process the data (i.e. clip low frequency words), the transformations outlined above seemed to produce good
representations for the models.

## Models
Here are the models I tried, and the intuition/concept outlines for each one. 
### NB Classifier
First, I tried using a Naive Bayes classifier, and it performed remarkably well! NB Classifiers are commonly used
in text classification tasks, and it was even more preferable here since we were dealing with small 2 to 3 character
hex chunks as opposed to actual words. Given a tokenized BOW representation of a binary blob, we can produce 
probabilities across the 12 classes by applying Bayes theorem.

Note that an independence assumption is made - despite that, however, a NB classifier seems to 
produce good results for this task probably because the blobs aren't as semantically complex as English language, 
and the data is probably clustered nicely.

### Logistic Regression + Linear SVM

For this task, I used a linear SVM as I predicted
the 12 ISAs are probably linearly separable. If I saw many misclassifications on the dev set with a linear classifier, I would
have tried finding a non-linear classifier (i.e. SVM with kernel or neural network), but these 2 models performed
reasonably well.

First, I tried implementing a one vs. all logistic regression classifier, and this performed decently well.  

Linear SVMs are also suitable for text classification tasks, so I tried to implement one for this. An SVM, or support
vector machine, is a supervised learning model that tries to separate data with a maximal margin. Unlike logistic
regression which finds some decently optimal decision boundary, SVMs strive to find the hyperplane that maximizes
the distance (or margin) between the nearest two training examples. After tuning hyperparameters for both models,
the linear SVM did provide slightly better results than LR.

### MLP

The final approach I tried was using a neural network. I implemented a standard feedforward network with 2 hidden layers.
The input layer took the BOW representation (potentially with tf-idf weighting applied) and the output layer
produced softmax probabilities across the 12 ISA classes. For this model, I used PyTorch instead of low-level
machine learning primitives as implementing backprop manually would have been a bit messy, and I had already 
found the NB Classifier and SVM to produce good results. However, I commented on all aspects of the implementation,
so please consult ```src\mlpmodel.py``` for further details. Note that this network doesn't perform 
very well, partially because of extremely sparse input tensors given by the short binary blobs and large vocabulary
sizes. I also did not optimize this model very well as the SVM and NB model were more favorable in this task.


