# encoding: utf-8
import os, sys, re
import collections
import numpy as np 

# import matplotlib
# matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
# from matplotlib import pyplot as plt

from pandas import DataFrame, Series
import pandas as pd 

###############################################################################################################
#
#   Module 
#   ------ 
#   Refactored from tpheno.seqmaker.seqUtils
#
#   Related 
#   -------
#   tpheno.seqmaker.seqSampling
# 
#
###############################################################################################################

def divide_interval(total, n_parts):
    pl = [0] * n_parts
    for i in range(n_parts): 
        pl[i] = total // n_parts    # integer division

    # divide up the remainder
    r = total % n_parts
    for j in range(r): 
        pl[j] += 1

    return pl 

def sample_class(X, y=None, n_samples=1000, replace=False, uniform=False): 
    """
    Input
    -----
    y: labels 
       use case: if labels are cluster labels (after running a clustering algorithm), then 
                 this funciton essentially performs a cluster sampling
    """
    N = len(X)
    if n_samples > N: 
        print(f"(sample_class) Warning: Requested sample size, {n_samples}, is greater than the total sample size N={N}")
    n_samples = min(N, n_samples)

    if y is None: 
       idx = np.random.choice(range(X.shape[0]), n_samples, replace=False)
       return X[idx], None

    assert X.shape[0] == len(y)
    labels = list(set(y))
    n_labels = len(labels)
    # print(f"(sample_class) labels: {labels}, y:\n{y}\n")

    if uniform: 
        n_subsets = divide_interval(n_samples, n_parts=n_labels) # 10 => [3, 3, 4]
        
        # [log] {0: 334, 1: 333, 2: 333}
        pdict = {labels[i]: n_subset for i, n_subset in enumerate(n_subsets)} # label -> subsample size
    else: 
        # sample classes proprotionally to their sizes
        pdict = collections.Counter(y)
        for label, count in collections.Counter(y).items(): 
            pdict[label] = int(n_samples * count/(N+0.0))
        Np = sum(pdict.values())
        N_delta = N - Np
        # print(f"... N_delta: {N_delta}")
        while N_delta > 0: 
            label = np.random.choice(labels, 1)[0]
            pdict[label] += 1
            N_delta -= 1
        assert sum(pdict.values()) == N, f"Class sizes summed to: {sum(pdict.values())} but N={N}"

    # print('verify> label to n_samples pdict:\n%s\n' % pdict) # ok. [log] {0: 500, 1: 500}
    tsx = []

    Xs, ys = [], []  # candidate indices
    for l, n in pdict.items(): 
        cond = (y == l)
        Xl = X[cond]
        # yl = y[cond]

        # sampling with replacement so 'n' can be larger than data size
        # idx = np.random.randint(Xl.shape[0], size=n)
        idx = np.random.choice(Xl.shape[0], size=n, replace=replace)
        # print('verify> select %d from %d instances' % (n, Xl.shape[0]))

        # print('verify> selected indices (size:%d):\n%s\n' % (len(idx), idx))
        Xs.append(Xl[idx, :])  # [note] numpy append: np.append(cidx, [4, 5])
        ys.append([l] * len(idx))
    
    assert len(Xs) == n_labels
    Xsub = np.vstack(Xs)  
    assert Xsub.shape[0] == n_samples   
    ysub = np.hstack(ys)
    assert ysub.shape[0] == n_samples

    return (Xsub, ysub)

def negative_sample(target_set, candidate_set, n=None, replace=False):
    """
    Subsample `n` elements from a candidate set (`candidate_set`) that 
    do not appear in the target set (`target_set`)
    """
    if not isinstance(target_set, (set, list, np.ndarray)): 
        target_set = set([target_set])
    if n is None: n = len(target_set)
    
    not_target_set = set(candidate_set)-set(target_set)
    N = len(not_target_set)
    if not replace: 
        assert N >= n 

    return np.random.choice(list(set(candidate_set)-set(target_set)), n, replace=replace)

def demo_class_sampling(): 
    from sklearn import datasets

    # [note] n_classes * n_clusters_per_class must be smaller or equal 2 ** n_informative
    X, y = datasets.make_classification(n_samples=3000, n_features=20,
                                    n_informative=15, n_redundant=3, n_classes=3,
                                    random_state=42)
    n_labels = len(set(y))
    print('data> dim(X): %s, y: %s > n_labels: %d' % (str(X.shape), str(y.shape), n_labels))

    Xsub, ysub = sample_class(X, y=y, n_samples=5000, replace=True)
    n_labels_sampled = len(set(ysub))
    print('sampled> dim(X): %s, y: %s > n_labels: %d' % (str(Xsub.shape), str(ysub.shape), n_labels_sampled))

    Xsub, ysub = sample_class(X, n_samples=5000, replace=True)
    n_labels_sampled = len(set(ysub))
    print('sampled> dim(X): %s, y(DUMMY): %s > n_labels: %d' % (str(Xsub.shape), str(ysub.shape), n_labels_sampled))

    return

def test(): 

    # negative sampling 
    subset = negative_sample([1, 3], list(range(10)))
    print(f"[test] subset:\n{subset}\n")

    # class sampling

    return

if __name__ == "__main__": 
    test()


