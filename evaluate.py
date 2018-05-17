# -*- coding: utf-8 -*-
'''
ischwaninger
01 April 2018

Standard evaluation metrics for a classifier performance based on Tweet2Vec implementation by bdhingra
https://github.com/bdhingra/tweet2vec
TODO: get other metrics
'''
import numpy as np


def precision(p, t, k):
    '''
    Compute precision @ k for predictions p and targets t
    '''
    n = p.shape[0]
    res = np.zeros(n)
    # for each prediction
    for idx in range(n):
        index = p[idx, :k]
        for i in index:
            if i == t[idx]:
                res[idx] += 1
    return np.sum(res) / (n * k)

#compute recall for target tx with predictions p and targets t
def recall(p, t, tx):

    #count targets
    tp = 0 
    fn = 0 
    for i in range (0, len(p)):
        if t[i] == tx and p[i] == tx:
            tp += 1
        elif t[i] == tx and p[i] != tx:
            fn += 1
    if (tp + fn) == 0.0:
        return 0.0
    return (tp / (tp + fn))
