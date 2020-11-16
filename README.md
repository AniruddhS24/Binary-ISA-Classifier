# PraetorianML

This is my attempt for the Praetorian Machine Learning Binaries (MLB) challenge. Here,
I will briefly describe my approach and choice of models. While ML concepts will be 
discussed here, please consult the code to see comments/documentation about specific
implementation and design choices. 

## Understanding the Task

The task at hand is: given a random binary blob, predict its instruction set architecture
(ISA). Essentially, we need to find a function that maps binary blobs to one of the 12
ISAs provided. Framing this as a supervised learning task, we can let our inputs be
some representation of a binary blob while the output is a class label for the ISA.

## Data Processing

All data processing routines can be found in ```src/dataprocessing.py``` where I generate
datasets, process input, and visualize data. 

Using the server, we can build our dataset by simply querying for random binary blobs to use as 
training examples. Since we can access the target architecture, we now have a fully labeled dataset.
Now, each training example is a (binary blob, target architecture) pair.
Moreover, to keep our data balanced with respect to the 12 classes, we generate an equal number of 
training examples from each class.

Upon some examination, I found that this is essentially a text classification problem -
first, we can convert our binary blobs to hexadecimal, and we can tokenize each blob by
chunks of hex characters. This was also suggested in the MLB article, so it seemed like a good
starting point. Viewing this challenge from an NLP setting now, we can create features
by extracting n-grams as sequences of hex characters (i.e. 'c30' or '17e'). The vocabulary will then be all of the
words (hex chunks) present in the training data.

Moreover, for each binary blob training example, we can simply represent it as an array of 'word' frequencies.
Our array size will be equal to the length of the vocabulary, and the relative order of words is not preserved - 
this is the standard bag-of-words (BOW) representation we will use. A BOW model would be suboptimal for many real-world NLP tasks - however, for this 
type of text classification task, I found that it's sufficient. 

However, storing raw counts poses an issue - upon examining the distribution of counts, some hex chunks such as
'00', '000', or '10' were extremely common across all ISA classes. These features are then probably not very useful,
yet they are assigned a high raw count value, so the model perceives them as important. To fix this, a common NLP 
technique is to use tf-idf weighting. 

![equation](https://latex.codecogs.com/gif.latex?tf%28t%2Cd%29%20%3D%20count%28t%2C%20d%29/%5Csum_%7Bt%27%20%5Cin%20d%7D%20count%28t%27%2C%20d%29%5C%5C%20idf%28t%2C%20D%29%20%3D%20%5Clog%7B%5Cfrac%7BN%7D%7B%7Cd%20%5Cin%20D%20%3A%20t%20%5Cin%20d%7C%7D%7D%20%5C%5C%20%5C%5C%20tfidf%28t%2C%20d%2C%20D%29%20%3D%20tf%28t%2Cd%29%20%5Ccdot%20idf%28t%2C%20D%29)

So yea