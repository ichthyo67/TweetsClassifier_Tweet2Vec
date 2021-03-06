# -*- coding: utf-8 -*-
'''
ischwaninger
25 Mar 2018

All networks parameters are stored here
based on Tweet2Vec implementation by bdhingra https://github.com/bdhingra/tweet2vec
'''

#input
NTWEETS = 10000 #number of tweets

# network size
CHAR_DIM = 10000  # dimensionality of the character embeddings lookup
MAX_CHAR = 8500  # number of unique characters (128 US-ASCII + 1,920 UTF-8 not including chinese chars)
HDIM = 500  # size of the hidden layer
LDIM = 2  # number of the unique output labels (categories)
MAX_LENGTH = 280 # max sequence of characters length for the input layer
MIN_LENGTH = 3 #min sequence of characters 
# twitter, weibo limits 140/280

# training parameters
SCALE = 0.1  # Initialization scale
BIAS = False  # use bias
NUM_EPOCHS = 1 # Number of epochs
N_BATCH = 64# 64 Batch size
LEARNING_RATE = 0.2 
MOMENTUM = 0.5 #default 0.5
REGULARIZATION = 0.0001
SCHEDULE = True  # use schedule
GRAD_CLIP = 5.  # gradient clipping for regularization

# thresholds
T1 = 0.01  # in learning schedule: if change < T1
T2 = 0.0001  # stopping criterion: if sum(deltas) / len(deltas) < T2
TRSH_CLSSFCN = 0.95 # label threshold: if 0.8, then classification only if 80% sure

# logging and model back ups
DISPF = 5  # Display frequency
SAVEF = 1000  # Save frequency
MODEL_PATH = "./model"
