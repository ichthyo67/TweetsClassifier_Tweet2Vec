# -*- coding: utf-8 -*-
'''
ischwaninger
01 April 2018

Load tweets from MongoDB and train Tweet2Vec NN model
'''
#from load_from_mongo import load_data_from_mongo_balanced, load_short_data_from_mongo
from load_from_sqlite import load_data_from_sqlite
from train import split_dataset, train_model
from inference import test_model
from settings import *
import sys

#Start Neural Network: load data, split data, run Tweet2Vec RNN, test best model
def train_protest_classifier():
    # load wicked tweets
    X1, y1 = load_data_from_sqlite("Tweets_Protest_Random_Test.db", "ProtestTweets", MAX_LENGTH, MIN_LENGTH, 1, random="true")

    assert X1
    assert y1
    print(len(X1), 'wicked samples loaded')

    # load random tweets
    X2, y2 = load_data_from_sqlite("Tweets_Protest_Random.db", "RandomTweets", MAX_LENGTH, MIN_LENGTH, NTWEETS, random="true")
    assert X2
    assert y2
    print(len(X2), 'random samples loaded')

    # merge data
    X = X1 + X2
    y = y1 + y2
    print(len(X), 'samples total')

    # split data
    assert len(X) == len(y)
    (X_train, y_train), (X_validate, y_validate), (X_test, y_test) = split_dataset(X, y)

    # run Tweet2Vec NN
    #train_model(X_train, y_train, X_validate, y_validate)
    test_model(X_test, y_test)


if __name__ == '__main__':
    if (len(sys.argv) > 1):
        print(sys.argv)
        NTWEETS = int(sys.argv[1])
        MAX_CHAR = int(sys.argv[2])
        CHAR_DIM = int(sys.argv[3])
        MAX_LENGTH = int(sys.argv[4])
        MIN_LENGTH = int(sys.argv[5])
        N_BATCH = int(sys.argv[6])
        LEARNING_RATE = float(sys.argv[7])
    train_protest_classifier()
