# -*- coding: utf-8 -*-
'''
ischwaninger
18 April 2018

Load tweets from database and train Tweet2Vec NN model
'''
#from load_from_mongo import load_data_from_mongo_balanced, load_short_data_from_mongo
from load_from_sqlite import load_data_from_sqlite
from train import split_dataset, train_model
from inference import test_model
from settings import *
import sys

'''
#mongo
def get_labeled_data():
    X, y = load_labeled_data_from_sqlite_balanced("Tweets.db", "Tweets",
                                         x_field="clean_text", y_field="label", limit=75000)

    assert X
    assert y
    print(len(X), 'samples loaded')

    (X_train, y_train), (X_validate, y_validate), (X_test, y_test) = split_dataset(X, y)
    train_model(X_train, y_train, X_validate, y_validate)
    test_model(X_test, y_test)

#mongo test
def test_get_random_tweets():
    X, y = load_data_from_mongo_balanced("tweets", "sample_04_12_2017",
                                         x_field="clean_text", y_value="random", limit=30)
    assert X
    assert y
    print(len(X), 'samples loaded')
    #print(X[0])
    #print(y[0])
'''

#Start Neural Network: load data, split data, run Tweet2Vec RNN, test best model
def train_protest_classifier():
    # load wicked tweets
    X1, y1 = load_data_from_sqlite("Tweets_Protest_Random.db", "ProtestTweets", MAX_LENGTH, MIN_LENGTH, NTWEETS, random=True)

    assert X1
    assert y1
    print(len(X1), 'wicked samples loaded')

    # load random tweets
    X2, y2 = load_data_from_sqlite("Tweets_Protest_Random.db", "RandomTweets", MAX_LENGTH, MIN_LENGTH, NTWEETS, random=True)
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

def test_run() :


    fruit = ["berry", "apple", "apple-plum", "plum", "I love berry", "banana in the fruitbasket", "my applejuice", "apple_ and banana", "berry", "wild berry", "little berries that are really really sour", "my other pear", "pears are better than apples", "banana", "pear", "plum with ebrry", "pear ice cream", "what is your favourite fruit", "favourite FRUIT", "oh know this must be a pluM", "oh this tastes good - a FRUIT", "speaking of fruits, what about vegetables?", "is an apple a fruit ", "what's wrong with berries", "apple juice sweet", "sweet berry ice cream", "berry with sugar", "sweet apples", "ice-cream without sugar", "fruit salad without sugar..."]
    veggy = ["tomato", "pepper", "chilli", "tomato with chilli", "chilli with rice", "sour veggies taste good", "is tomato a fruit or a veggy", "favourite veggy?", "little tomato vs big tomato", "red pepper or green pepper", "favourite color of chilli?", "plum or tomato? is my VEGGY concept", "pears with salad and appe and chili", "my tomatojuice", "juice made of veggy and sugar is VEGGY", "veggy juie", "veggy ice cream is ALWAYS VEGGY", "tomato salad", "tomato salad with some chilli added", "tomato pepper", "pepper tomato", "chilli with more chilli", "sour chilli", "veggy for veggies", "green red for veggy", "tomato sauce with noodles", "noodles or rice for tomato sauce?", "sauce out of chilli", "sandwich with veggies", "nice veggy salad you have!!"]
    print(len(fruit))
    print(len(veggy))

    X1 = fruit[0:NTWEETS-1]
    y1 = []
    X2 = veggy[0:NTWEETS-1]
    y2 = []

    for i in range(0, NTWEETS-1):
        y1.append("fruit")
        y2.append("veggy")


    assert X1
    assert y1
    assert X2
    assert y2
    assert len(X1) == len(y1)
    print(len(X1), 'fruit samples loaded')
    print(len(X2), 'veggy samples loaded')

    # merge data
    X = X1 + X2
    y = y1 + y2

    print(len(X), 'samples total')

    # split data
    assert len(X) == len(y)
    (X_train, y_train), (X_validate, y_validate), (X_test, y_test) = split_dataset(X, y)
    train_model(X_train, y_train, X_validate, y_validate)
    test_model(X_test, y_test)


if __name__ == '__main__':
    if (len(sys.argv) > 1):
        print(sys.argv)
        NTWEETS = int(sys.argv[1])
        CHAR_DIM = int(sys.argv[2])
        MAX_LENGTH = int(sys.argv[3])
        MIN_LENGTH = int(sys.argv[4])
        N_BATCH = int(sys.argv[5])
        LEARNING_RATE = float(sys.argv[6])
    train_protest_classifier()
    #test_run()
