# -*- coding: utf-8 -*-
'''
ischwaninger / svakulenko
03 March 2018

Run inference on the pretrained model based on Tweet2Vec implementation by bdhingra: test_char.py
https://github.com/bdhingra/tweet2vec
'''
from collections import OrderedDict
import pickle as pkl

import statistics

import numpy as np
import lasagne
import theano
import theano.tensor as T

from settings import *
from train import BatchTweets, prepare_data
from tweet2vec import tweet2vec
from evaluate import precision, recall


def load_params(path):
    """
    Load previously saved model
    """
    params = OrderedDict()

    with open(path,'rb') as f:
        npzfile = np.load(f)
        for kk, vv in npzfile.iteritems():
            params[kk] = vv

    return params


def classify(tweet, t_mask, params, n_classes, n_chars):
    # tweet embedding
    emb_layer = tweet2vec(tweet, t_mask, params, n_chars)
    # Dense layer for classes
    l_dense = lasagne.layers.DenseLayer(emb_layer, n_classes, W=params['W_cl'], b=params['b_cl'], nonlinearity=lasagne.nonlinearities.softmax)

    return lasagne.layers.get_output(l_dense)


def test_model(Xt, yt, model_path=MODEL_PATH):

    # Load model and dictionaries
    print("Loading model params...")
    params = load_params('%s/best_model.npz' % model_path)
    print("Loading dictionaries...")
    with open('%s/dict.pkl' % model_path, 'rb') as f:
        chardict = pkl.load(f)
    with open('%s/label_dict.pkl' % model_path, 'rb') as f:
        labeldict = pkl.load(f)

    n_char = len(chardict.keys()) + 1
    n_classes = len(labeldict.keys())
    print("Labeldict:", labeldict)
    print("Building network...")

    # Tweet variables
    tweet = T.itensor3()
    targets = T.imatrix()

    # masks
    t_mask = T.fmatrix()

    # network for prediction
    predictions = classify(tweet, t_mask, params, n_classes, n_char)

    # Theano function
    print("Compiling theano functions...")
    predict = theano.function([tweet,t_mask], predictions)

    # Test
    print("Testing...")

    # iterator over batches
    #test on entire set -> iterate, and memorize
    precision_mean = 0
    test_iter = BatchTweets(Xt, yt, labeldict, batch_size=N_BATCH)
    #for i in range (0, len_batches):
    n_testsamples = 0
    maxprec = 0.
    precisions = []
    recalls0 = []
    recalls1 = []
    #corrections with new threshold
    count_corrections = 0
    classified_1 = 0 #how many were classified in total?
    for xr, y in test_iter:
        n_testsamples +=len(xr)
        preds = []
        targs = []

        #xr, y = testBatches[i] #list
        x, x_m = prepare_data(xr, chardict, n_chars=n_char)
        vp = predict(x, x_m)
        #print("VP", vp)
        #np.argsort returns [0, 1] or [1, 0] according to vp floats
        #TODO: add weights to certain characters here or earlier?
        ranks = np.argsort(vp)[:, ::-1]
        #print("RANKS", ranks)
        for idx, item in enumerate(xr):
            corr = False
            if ranks[idx,:][0] == 1:
                if vp[idx][1] < TRSH_CLSSFCN:
                    ranks[idx,:][0] = 0
                    ranks[idx,:][1] = 1
                    count_corrections += 1
                    corr = True
                else:
                    classified_1 += 1
            preds.append(ranks[idx,:])
            targs.append(y[idx])
            #print("Xt LEN", len(Xt), "INDEX: ", idx)
            if not corr:
                print(xr[idx], ranks[idx,:][0], y[idx])
            else:
                print(xr[idx], ranks[idx,:][0], y[idx], "correction due to", vp[idx])
            #print(Xt[idx], ranks[idx][0], y[idx])
        print("Predictions", [ranks[0] for ranks in preds])
        print("Targets", y)

        # compute precision @1
        validation_cost = precision(np.asarray(preds), targs, 1)
        print(validation_cost)

        rec0 = recall([ranks[0] for ranks in preds], targs, 0)
        recalls0.append(rec0)

        rec1 = recall([ranks[0] for ranks in preds], targs, 1)
        recalls1.append(rec1)
        print("recall protest", rec1) #p, t, tx
        print("recall random", rec0)

        #for standard deviation
        precisions.append(validation_cost)

        if validation_cost > maxprec:
            maxprec = validation_cost
        precision_mean += validation_cost*len(xr)

        # print [labeldict.keys()[rank[0]] for rank in ranks]
    print()
    print("#"*10, "FINAL OUTPUT", "#"*10)
    print()
    print("%d Tweets tested"%n_testsamples)
    print("%d classified 1"%classified_1)
    print("corrections with higher threshold", TRSH_CLSSFCN, ":", count_corrections)
    print("Mean Precision weighted", precision_mean / n_testsamples)
    print("Max Precision on Batch", maxprec)
    #standarddeviation = sum((-precision_mean + [prec for prec in precisions])) * 1/(1-n_testsamples)
    print("Mean Precision unweighted", statistics.mean(precisions))
    print("Standard Deviation", statistics.stdev(precisions))

    print("Mean Recall 0", statistics.mean(recalls0), "Stdev", statistics.stdev(recalls0))
    print("Mean Recall 1", statistics.mean(recalls1), "Stdev", statistics.stdev(recalls1))

    print("#" * 20)



def test_test_model():
    # test generalization performance of the model
    X = ["hot", "hot and ", "ho", "cold"]
    y = ["hot", "hot", "hot", "cold"]
    test_model(X, y)


if __name__ == '__main__':
    # test_infer()
    test_test_model()
