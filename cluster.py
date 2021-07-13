# encoding: utf-8

import sys, os
import re, string
import collections
import random, math 

from pandas import DataFrame, Series
import pandas as pd
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle

import matplotlib.cm as cm
import seaborn as sns

import matplotlib
# matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt

# select plotting style 
plt.style.use('seaborn')  # values: {'seaborn', 'ggplot', }

import utils, learn_manifold
from utils import div
### clustering algorithms 
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering, DBSCAN
# from sklearn.cluster import KMeans

# from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# from sklearn.cluster import AffinityPropagation

# evaluation 
from sklearn import metrics
from scipy.spatial import distance
# import scipy.spatial.distance as ssd  

## default IO
class Analysis: 
    prefix = os.getcwd()
    data_dir = os.path.join(prefix, 'cluster_analysis')

"""


    Reference
    ---------

    1. clustering visualization 

        https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

    2. 
"""

def map_clusters(y_pred, items=[], relabeling=True, verbose=0):
    """
    Map cluster/class labels to cluster IDs. 
    This groups items into clusters according to the cluster labels (i.e. y_pred, a list of cluster IDs)

    Read the labels array and clusters label and return the set of words in each cluster

    Input
    -----
    y_pred: cluster IDs ~ oredering the training instances (X)
    item: a representation that corresponds to the true label (e.g. y_true, y_tags)

    Output
    ------
    A hashtable: cluster IDs -> labels
   
    e.g. 

    data = [x1, x2, x3, x4 ]  say each x_i is a 100-D feature vector
    labels = [0, 1, 0, 0]  # ground truth labels
    clusters = [0, 1, 1, 0]   say there are only two clusters 
    =>  
       map_clusters given labels

       {0: [0, 0], 1: [1, 0]}

       map_clusters given data 

       {0: [x1, x4], 1: [x2, x3]}

       map_clusters withou items as input 

       {0: [0, 3], 1: [1, 2]}  # positional indices

    """
    # cluster_to_items = utils.autovivify_list()
    cluster_to_items = {cid: [] for cid in np.unique(y_pred)}
       
    # mapping i-th label to cluster cid
    if len(items) > 0: 
        tUpToPermunation = False
        for i, cid in enumerate(y_pred):  
            cluster_to_items[cid].append( items[i] )
            if type(cid) == type(items[i]): 
                tUpToPermunation = True

        if relabeling and tUpToPermunation:  
            cluster_to_items2 = {cid: [] for cid in np.unique(y_pred)}
            
            for cid, members in cluster_to_items.items(): 
                label_counts = collections.Counter(members)
                majority_label = label_counts.most_common(1)[0][0]
                cluster_to_items2[majority_label] = members  

            # but it only makes sense to relabel by majority vote when this majority can be uniquely identified for each cluster
            if len(cluster_to_items2) == len(cluster_to_items): 
                if verbose: 
                    print("(map_clusters) Cluster IDs re-assigned to the majority")
                cluster_to_items = cluster_to_items2
    else: 
        # when item labels are unknown, then we simply construct a map from cluster IDs to positional indices ...
        # ... assuming that each data point has its own 'label'
        for i, cid in enumerate(y_pred):  
            cluster_to_items[cid].append( i )

    return cluster_to_items

def merge_cluster(clusters, cids=[]): 
    """
    From multibag represention back to cluster ID sequence representation. 
    
    where, 

    multibag repr maps cluster IDs to members 
    cluster ID sequence is a list of cluster IDs 


    """
    assert isinstance(clusters, dict)
    member = clusters.itervalues().next()
    assert hasattr(member, '__iter__')

    data = []
    if len(cids) == 0: # merge all 
        for cid, members in clusters.items(): # members can be a list or list of lists
            data.extend(members)
    else: 
        for cid in cids: 
            try: 
                members = clusters[cid]
                data.extend(members)
            except: 
                pass
    if not data:
        assert cids is not None 
        print('warning> No data selected given cids:\n%s\n' % cids)
    return data
### alias 
multibag_to_ids = merge_cluster

def summarize_cluster(y_pred, X, y=None, topk=10, 
                            distance_metric=distance.cosine, cluster_method='?', verbose=0):  # [refactor] also see cluster.analyzer
    # from sklearn import metrics
    # rom scipy.spatial import distance

    if verbose: highlight(message=f'(summarize_cluster) topk: {topk}, distance metric: {distance_metric}, clustering: {cluster_method}')
    clusters = map_clusters(y_pred, X)  # id -> {X[i]}
    
    res = {}
    # compute mean and medians [todo] can be more efficient
    cluster_means, cluster_medians, cluster_medoids = {}, {}, {}
    for cid, points in clusters.items(): 
        cluster_means[cid] = np.mean(points, axis=0)
        cluster_medians[cid] = np.median(points, axis=0)

    nrows = len(y_pred)
    assert X.shape[0] == nrows
    c2p = map_clusters(y_pred, range(nrows))  # cluster to position i.e. id -> indices 
    
    # k-nearest neighbors wrt mean given a distance metric 
    cluster_knn = {}
    for cid, mpoint in cluster_means.items(): # foreach cluster (id) and its centroid
        idx = c2p[cid]  # idx: all data indices in cluster cid 
        rankedDist = sorted([(i, distance_metric(X[i], mpoint)) for i in idx], key=lambda x: x[1], reverse=True)[:topk]
        # if cid % 2 == 0: print('test> idx:\n%s\n\nranked distances:\n%s\n' % (idx, str(rankedDist)))
        cluster_knn[cid] = [i for i, d in rankedDist]

    cluster_knn_median = {}
    for cid, mpoint in cluster_medians.items(): # foreach cluster (id) and its centroid
        idx = c2p[cid]  # idx: all data indices in cluster cid 
        rankedDist = sorted([(i, distance_metric(X[i], mpoint)) for i in idx], key=lambda x: x[1], reverse=True)[:topk]
        # if cid % 2 == 0: print('test> idx:\n%s\n\nranked distances:\n%s\n' % (idx, str(rankedDist)))
        cluster_knn_median[cid] = [i for i, d in rankedDist]

    # save statistics 
    res['cluster_means'] = cluster_means  # cid -> cluster centroid (mean)
    res['cluster_medians'] = cluster_medians  # cid -> cluster centroid (median)
    res['cluster_knn'] = cluster_knn   # knn wrt mean; cid -> KNN members (wrt mean) in X's positional IDs
    res['cluster_knn_median'] = cluster_knn_median  # knn wrt median; 

    return res

def evaluate_cluster(y_pred, y=None, X=None, cluster_method='?'): 
    import sampling_utils as sampling
    # from sklearn import metrics
    # rom scipy.spatial import distance

    y_true = y
    div(message='(evaluate_cluster) clustering algorithm: %s' % cluster_method)

    mdict = {}
    if y_true is not None: 
        # A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class.
        # ... This metric is independent of the absolute values of the labels
        mdict['homogeneity'] = metrics.homogeneity_score(y_true, y_pred)
        print("Homogeneity: %0.3f" % mdict['homogeneity'])

        # A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster.
        # ... This metric is independent of the absolute values of the labels
        mdict['completeness'] = metrics.completeness_score(y_true, y_pred)
        print("Completeness: %0.3f" % mdict['completeness'])

        # The V-measure is the harmonic mean between homogeneity and completeness
        mdict['v_measure'] = metrics.v_measure_score(y_true, y_pred)
        print("V-measure: %0.3f" % mdict['v_measure'])

        # The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples 
        # and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings
        mdict['ARI'] = metrics.adjusted_rand_score(y_true, y_pred)
        print("Adjusted Rand Index: %0.3f" % mdict['ARI'])

        # Adjusted Mutual Information (AMI) is an adjustment of the Mutual Information (MI) score to account for chance. 
        # It accounts for the fact that the MI is generally higher for two clusterings with a larger number of clusters
        mdict['AMI'] = metrics.adjusted_mutual_info_score(y_true, y_pred)
        print("Adjusted Mutual Information: %0.3f" % mdict['AMI'])
     
    if X is not None: 
        div(message='... Evaluating the clustering withtout the ground truth (y_true)')

        # only silhouette coeff doesn't require ground truths (but it expensive to compute) => consider subsamping
        n_samples = min(X.shape[0], 1000) 
        try: 
            Xsub, ysub = sampling.sample_class(X, y=y_pred, n_samples=n_samples, replace=False) # [todo] without replacemet => replacement upon exceptions
        except: 
            print('cluster.cluster_analysis> could not sample X without replacement wrt cluster labels (dim X: %s while n_clusters: %d)' % \
                (str(X.shape), len(set(y_pred)) ))
            Xsub, ysub = sampling.sample_class(X, y=y_pred, n_samples=n_samples, replace=True)

        assert Xsub.shape[1] == X.shape[1], "Input X has %d vars while sampled subset has %d vars" % (X.shape[1], Xsub.shape[1])
        try: 
            mdict['silhouette'] = metrics.silhouette_score(Xsub, np.array(ysub), metric='sqeuclidean')
            print("Silhouette Coefficient: %0.3f" % mdict['silhouette'])
        except Exception as e: 
            # [log] could not broadcast input array from shape (1000,100) into shape (1000)
            print('cluster.cluster_analysis> Could not compute silhouette score: %s' % e)
        
    return mdict 

def cluster_sampling(X, y_pred, n_samples=None, **kargs):
    import sampling_utils as sampling

    if n_sample is None: n_sample = max(1, X.shape[0]/10)

    Xsub, cids = sampling.sample_class(X, y=[], n_samples=n_samples, **kargs) 

    return (Xsub, cids)

def eval_knn(cluster_repr, topk=10, metric=distance.cosine): 
    """
    Input
    -----
    cluster_repr: cluster representatives in terms of a dictionary: cid -> representative point (e.g. centroid)
    topk: k in knn
 
    """
    cluster_knn = {}
    for cid, rpt in cluster_repr.items(): # foreach cluster (id) and its centroid
        idx = c2p[cid]  # idx: all data indices in cluster cid 
        rankedDist = sorted([(i, metric(X[i], rpt)) for i in idx], key=lambda x: x[1], reverse=True)[:topk]
        # if cid % 2 == 0: print('test> idx:\n%s\n\nranked distances:\n%s\n' % (idx, str(rankedDist)))
        cluster_knn[cid] = [i for i, d in rankedDist]
    return cluster_knn

def cluster_analysis(X, y=None, model=None, **kargs): 
    """

    Params 
    ------
    X 
    y: class labels (often in integer notation) 
    y_tags: class tags/annotations; a tag can be any string associated with a class ID such as the meaning of 
            a given class

    Output
    ------

       Plots
            cluster_distribution

    """
    def cluster_distribution(ldmap, cluster_names=None, plot_dir='output'): # ldmap: label to distribution map
        
        # plot histogram of distances to the origin for all document vectors
        for cid, distr in ldmap.items():  
            print(f"... Cluster ID: {cid}, distribution: {distr}")
            
            plt.clf()  # plot for each cluster ...
            canonical_label = cluster_names[cid] if cluster_names is not None else ''
            
            f = plt.figure(figsize=(8, 8))
            sns.set_theme(style="whitegrid")
            sns.set(rc={"figure.figsize": (8, 8)})
            
            # sns.distplot(distr, bins=intervals)
            sns.barplot(x=cluster_names, y=distr)

            identifier_distr = 'L%s-P%s' % (canonical_label, identifier) if canonical_label else 'P%s' % identifier
            fpath = os.path.join(plot_dir, 'cluster_distribution-%s.tif' % identifier_distr)
            print('[output] saving cluster distribution to %s' % fpath)
            plt.savefig(fpath)

        return

    from utils import highlight
    # from sklearn.cluster import AgglomerativeClustering
    # from sklearn.neighbors import kneighbors_graph
    # from sklearn.cluster import AffinityPropagation
    # import sampling

    # general paramters
    experiment_id = kargs.get('experiment_id', 'zero-shot') 
    
    # clustering parameters: n_clusters, n_classes, cluster_method, class_tags
    ########################################################  
    verbose = kargs.get('verbose', 0)  # verbosity level
    n_clusters = kargs.get('n_clusters', 5)
    # n_classes = n_clusters
    n_clusters_est = -1

    # specify clustering method via string; relevant only `model` is not given (i.e. `model` is None)
    cluster_method = kargs.get('cluster_method', model if isinstance(model, str) else str(model))
    optimize_k = kargs.get('optimize_k', False)
    run_tsne = kargs.get('run_tsne', False)

    var_clusters = True if cluster_method.startswith(('agg', 'aff', 'db', 'hdb')) else False
    
    # from cluster labels to meanings
    y_tags = kargs.get('y_tags', {})  # y_tags maps from class labels to tags
    y_pred = kargs.pop('y_pred', None) # clustering output from the pre-trained model

    # I/O parameters
    #########################################################
    output_dir = kargs.get('output_dir', os.path.join(os.getcwd(), 'output')) 
    if not os.path.exists(output_dir): os.makedirs(output_dir) # base directory
    save_cluster = kargs.get('save_', True)  # save cluster labels? 
    test_cluster = kargs.get('test_', True)
    plot_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(plot_dir): os.makedirs(plot_dir) # base directory

    identifier = kargs.get('identifier', '%s-C%s-nC%s' % (experiment_id, cluster_method, n_clusters))
    if var_clusters: identifier = kargs.get('identifier', '%s-C%s' % (experiment_id, cluster_method))
    
    if run_tsne: identifier += 'Mtsne'

    # load pre-computed clustering results
    load_path = os.path.join(output_dir, 'cluster-%s.csv' % identifier)  
    load_cluster = kargs.get('load_', True) and os.path.exists(load_path)
    #########################################################

    # Data parameters
    #########################################################
    Xid = kargs.get("Xid", np.array(range(0, X.shape[0])))
    doc_labeled = doc_tagged = False
    if y is None: y = np.repeat(1, X.shape[0]) # unlabeled data
    n_classes = len(set(y))
    if n_classes > 1: 
        doc_labeled = True # each document is represented by its (surrogate) labels
    if y_tags is not None and len(y_tags) > 0: 
        doc_tagged = True
    ##########################################################

    # y_true = y
    highlight(f'(cluster_analysis) Method: {cluster_method}, Input shape: {X.shape})')
    print(f"... # of clusters requested: {n_clusters}, # of classes: {n_classes} (1 if `y` is unknown)")

    # preprocess with T-SNE or other manifold learning methods? 
    if run_tsne: 
        X = learn_manifold.tsne(X)

    # 1. Run Cluster Analysis 
    ##########################################################
    n_cluster_est = None   
    if not load_cluster: 
        if isinstance(model, str):  
            clustering, _ = run_cluster_analysis(X, y, model=model, n_clusters=n_clusters, output_dir=output_dir, optimize_k=False)
            y_pred = clustering.labels_
            # cluster_inertia   = model.inertia_
        else: 
            if y_pred is None: 
                clustering = model.fit_predict(X)
                try: 
                    y_pred = clustering.labels_  # scikit-learn methods should support this attribute 
                except: 
                    if isinstance(clustering, (np.ndarray, list)): 
                        assert len(clustering) == X.shape[0]
                        y_pred = clustering
                    else: 
                        raise ValueError(f"Could not infer the cluster labeling with model={model}")
            else: 
                assert len(y_pred) == X.shape[0]
    else: 
        print(f'[I/O] Loading pre-computed clustering (method={cluster_method}, ID={identifier})... ')
        df = pd.read_csv(load_path, sep=',', header=0, index_col=False, error_bad_lines=True)
        y_pred = df['cluster_id'].values
        assert X.shape[0] == len(y_pred), "nrow of X: %d while n_cluster_ids: %d" % (X.shape[0], len(y_pred))
    n_clusters_est = len(np.unique(y_pred))
    ##########################################################
    if verbose: highlight(f'(cluster_analysis) Completed clustering method: {cluster_method}')
    if not var_clusters: print(f"... n_clusters obtained: {n_clusters_est} =?= n_clusters expected: {n_clusters}")

    # 2. Run Post-hoc cluster analysis
    ##########################################################
    res_metrics = {}
    if y is not None and n_classes > 1: 
        res_metrics = evaluate_cluster(y_pred, X=X, y=y, cluster_method=cluster_method)

    res_summary = summarize_cluster(y_pred, X=X, y=y, topk=10, 
                        distance_metric=distance.cosine, cluster_method=cluster_method)
    ##########################################################

    # Test
    membership  = map_clusters(y_pred, y)  # y: pre-computed labels (e.g. heuristic labels)
    # print(f"... number of unique cluster labels: {len(np.unique(y_pred))}")
    # print(f"... membership:\n{membership}\n")

    if var_clusters: 
        # these clustering methods do not require n_clusters
        n_clusters = len(membership)
    else: 
        assert len(membership) == n_clusters, f"n_clusters={n_clusters} but running clustering gives {len(membership)} clusters"
    
    # 3. Save clustering results 
    if save_cluster: 
        header = ['cluster_id', ] # ['id', 'cluster_id', ]
        adict = {h:[] for h in header}
        fpath = os.path.join(output_dir, 'cluster_id-%s.csv' % identifier)

        for i, cl in enumerate(y_pred):
            # adict['id'].append(Xid[i])
            adict['cluster_id'].append(cl)

        df = DataFrame(adict, columns=header) 
        if doc_labeled:
            header = ['id', 'cluster_id', 'label', ]
            df['label'] = y
        if doc_tagged: 
            if isinstance(y_tags, dict): 
                df['tag'] = [y_tags[ye] for ye in y]
            else: 
                assert len(y_tags) == len(y)
                df['tag'] = y_tags

        print('[I/O] Saving cluster map (id -> cluster id) to %s' % fpath)
        df.to_csv(fpath, sep=',', index=False, header=True)  

        # [output]
        if membership is not None: 
            fpath = os.path.join(output_dir, 'cluster_map-%s.csv' % identifier)  
            header = ['cluster_id', 'data', ] # in general, this should include 'id'
            adict = {h:[] for h in header}
            size_cluster = 0
            for cid, members in membership.items():
                size_cluster += len(members)
            size_avg = size_cluster/(len(membership)+0.0)
            if verbose: print('[verify] Averaged %s-cluster size: %f' % (cluster_method, size_avg))

        # [output] save knn 
        if res_summary: 
            header = ['cluster_id', 'knn', ]
            cluster_knn_map = res_summary['cluster_knn']  # wrt mean, cid -> [knn_id]
            sep_knn = ','
            adict = {h: [] for h in header}
            for cid, knn in cluster_knn_map.items(): 
                adict['cluster_id'].append(cid) 
                adict['knn'].append(sep_knn.join([str(e) for e in knn]))
        
            fpath = os.path.join(output_dir, 'cluster_knnmap-%s.csv' % identifier)
            print('[I/O] Saving knn-to-centriod map (cid -> knn wrt centroid) to %s' % fpath)
            df = DataFrame(adict, columns=header)
            df.to_csv(fpath, sep='|', index=False, header=True)

    # Optimal number of clusters?
    if test_cluster: 
        if doc_labeled: 
            div(message='Testing cluster consistency with surrogate labels (%s) ...' % cluster_method)
            cluster_ids = set(y_pred)  
            ratios = {l:[] for l in cluster_ids}
            for cid_i in cluster_ids:  # foreach unique cluster ... 
                # ... find its label distribution (good clustering should result in labeling with high purity)
                label_counts = collections.Counter(membership[cid_i])
                cluster_size = sum(label_counts.values())
                for cid_j in cluster_ids:   # foreach cluster (cluster id -> members)
                    if cluster_size == 0: 
                        r = 0.0 
                    else: 
                        r = label_counts.get(cid_j, 0)/(cluster_size+0.0) # given a (true) label, find the ratio of that label in a cluster
                    ratios[cid_i].append(r)
            print('(cluster_analysis) cluster-label distribution (method: %s):\n%s\n' % (cluster_method, ratios))
 
            names = [y_tags[i] for i in cluster_ids] if doc_tagged else [f'label_{i}' for i in cluster_ids]
            cluster_distribution(ratios, cluster_names=names) # y_tags maps cluster IDs to their meaningful annotations/tags
            # ... Ideally, we should observe a very skewed distribution i.e. most clusters contain only few specific labels

    return (y_pred, res_metrics)  # cluster id to document members (labels or content)

def plot_cluster_distribution(ldmap, output_dir='', meta=''): # ldmap: label to distribution map
    def name_file(label='', ext='tif'): 
        fname = 'cluster_distribution_L{label}'.format(label=label) if label else 'cluster_distribution'  
        if meta: 
            fname = '%s_M%s' % (fname, meta) # e.g. method by which clusters are formulated
        fname = '%s.%s' % (fname, ext) 
        return fname
    # [params] testdir
    # plot histogram of distancesto the origin for all document vectors
    if not output_dir: output_dir = Analysis.data_dir
    
    for ulabel, distr in ldmap.items(): 
        plt.clf() 
        
        canonical_label = lmap.get(ulabel, '')
        
        f = plt.figure(figsize=(8, 8))
        sns.set(rc={"figure.figsize": (8, 8)})
        
        intervals = [i*0.1 for i in range(10+1)]
        sns.distplot(distr, bins=intervals)

        fpath = os.path.join(output_dir, name_file(label=ulabel))
        print('(plot_cluster_distribution) Saving cluster distribution to %s' % fpath)
        plt.savefig(fpath)

    return

def clusterDistribution(y_pred, labels, **kargs):
    """

    Memo
    ----
    1. find the label distributions in clusters 

       e.g. given a label, say 'has_disease' (vs control)
            say we have 3 clusters 

            ratios['has_disease']: [0.1, 0.2, 0.7]


    """
    # import collections
    # find the distribution of labels in clusters: foreach (true) label, find which clusters tend to incorporate that label

    cluster_to_labels  = map_clusters(y_pred, labels)

    ulabels = set(labels) # unique labels
    n_labels = len(ulabels)
    ratios = {l:[] for l in ulabels}

    cluster_ids = sorted(np.unique(y_pred))  # fix the ordering of cluster IDs
    for label in ulabels: 

        for cid in cluster_ids: 
            label_subset = cluster_to_labels[cid]   # label distribution in the cluster of ID = cid
            counts = collections.Counter(label_subset)
            r = counts[label]/(len(label_subset)+0.0) # given a (true) label, find the ratio of that label in a cluster
            ratios[label].append(r)  #  
        # each label in ratios should be mapped to a list of size n_clusters 

    cluster_method = kargs.get('method', '?')
    print('[clusterDistribution] cluster-label distribution (method: %s):\n%s\n' % (cluster_method, ratios))

    # maps each label-specific group to the label distrituion 
    # e.g. label = logistic_classifier => {cluster 1: 0.8, cluster 2: 0.1, cluster 3: 0.1} => 80% is in cluster 1
    output_dir = kargs.get('output_dir', Analysis.data_dir)
    plot_cluster_distribution(ratios, output_dir=output_dir, meta=cluster_method)  # ratios is a multimap

    return

def eval_linkage(X, **kargs): 
    import seaborn as sns
    from matplotlib import pyplot as plt 
    import scipy.spatial as sp
    import scipy.cluster.hierarchy as hc
    from PIL import Image 
    from matplotlib import cm
    from cluster_utils import evalSimilarity

    # X: either a dataframe (in column vector format) or a 2D array (in row vector format) 

    # A = X.T.values if isinstance(X, DataFrame) else X 
    # ... assuming that the input dataframe is in column vector form
    cols = []
    if isinstance(X, DataFrame): 
        A = X.T.values   # assuming that X is in column vector format
        cols = X.columns.values
    else: 
        A = X

    # A: in row vector format (each instance is represented by a row vector)
    S = evalSimilarity(A)
    print('... dim(S): {0}'.format(S.shape))
    D = 1. - S

    # turn into dataframes 
    if len(cols) == 0: cols = ['x_{0}'.format(i) for i in D.shape[0]]
    DX = DataFrame(D, columns=cols, index=cols)
    
    # Visualize distance matrices
    # img = Image.fromarray(Du, 'RGB')
    # img = Image.fromarray(np.uint8(cm.gist_earth(X)*255))
    # img.save('X_Dim{0}-{1}.tif'.format(X.shape[0], X.shape[1]), dpi=(500,500)) # os.path.join(prefix, 'wmf_Su.png')
    # saveFig(img, plot_path(name='wmf_Su'), dpi=300)

    # [note] diag(Du) must be all zeros, not even acceptable with very small numbers
    method = kargs.get('method', 'ward')
    try: 
        linkage = hc.linkage(sp.distance.squareform(DX), method=method)
    except Exception as e:
        print('(eval_linkage) Warning: %s' % e) 
        
        D = DX.values
        np.fill_diagonal(D, 0.0)
        DX = DataFrame(D, columns=cols, index=cols)

        linkage = hc.linkage(sp.distance.squareform(DX), method=method)
    return linkage

def run_cluster_map(X, **kargs): # method='ward', output_dir=None
    # Libraries
    import seaborn as sns
    from matplotlib import pyplot as plt 
    import scipy.spatial as sp
    import scipy.cluster.hierarchy as hc
    from PIL import Image 
    from matplotlib import cm
    from cluster_utils import evalSimilarity
    from utils import saveFig

    # X: either a dataframe (in column vector format) or a 2D array (in row vector format) 

    # A = X.T.values if isinstance(X, DataFrame) else X 
    # ... assuming that the input dataframe is in column vector form
    cols = []
    if isinstance(X, DataFrame): 
        A = X.T.values   # assuming that X is in column vector format
        cols = X.columns.values
    else: 
        A = X

    # A: in row vector format (each instance is represented by a row vector)
    S = evalSimilarity(A)
    print('... dim(S): {0}'.format(S.shape))
    D = 1. - S

    # turn into dataframes 
    if len(cols) == 0: cols = ['x_{0}'.format(i) for i in D.shape[0]]
    DX = DataFrame(D, columns=cols, index=cols)
    
    # Visualize distance matrices
    # img = Image.fromarray(Du, 'RGB')
    img = Image.fromarray(np.uint8(cm.gist_earth(X)*255))
    img.save('X_Dim{0}-{1}.tif'.format(X.shape[0], X.shape[1]), dpi=(500,500)) # os.path.join(prefix, 'wmf_Su.png')

    # [note] diag(Du) must be all zeros, not even acceptable with very small numbers
    method = kargs.get('method', 'ward')
    try: 
        linkage = hc.linkage(sp.distance.squareform(DX), method=method)
    except Exception as e:
        print('(run_cluster_map) Warning: %s' % e) 
        
        D = DX.values
        np.fill_diagonal(D, 0.0)

        # make symmetric 
        # D = .5 * (D + D.T)

        DX = DataFrame(D, columns=cols, index=cols)

        linkage = hc.linkage(sp.distance.squareform(DX), method=method)

    g = sns.clustermap(DX, row_linkage=linkage, col_linkage=linkage)

    output_dir = kargs['output_dir'] if 'output_dir' in kargs else os.path.join(os.getcwd(), 'plot')
    if not os.path.exists(output_dir):
        print('(run_cluster_map) Creating output directory: %s' % output_dir)
        os.mkdir(output_dir)
    file_name = kargs.get('file_name', 'clustermap_XD{0}-{1}_M{2}'.format(X.shape[0], X.shape[1], method))  
    SaveFig(g, plot_path(name=file_name, basedir=output_dir), dpi=kargs.get('dpi', 300))

    return g

def run_cluster_analysis(X, y=None, model='kmeans', **kargs):  
    """
    Main routine for running cluster analysis. 

    Params
    ------
       output_dir
       cluster_method
       identifier
       n_clusters


    Related
    -------
    cluster_analysis()
    run_silhouette_analysis() for kmeans

    """
    import gap_stats
    import bisect 

    # from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering, DBSCAN
    output_dir = kargs.get('output_dir', os.path.join(os.getcwd(), 'data'))
    identifier = kargs.get('identifier', 'CA')  # CA: cluster analysis

    # parameters for clustetring
    ###################################################
    cluster_method = model
    n_clusters = kargs.get('n_clusters', 10)
    n_clusters_est = None
    print('(run_cluster_analysis) Requested %d clusters' % n_clusters)
    maxK = min(X.shape[0]/2, 200)
    k_optimized = False 
    k_exempt = ('aff', 'db', )
    optimize_k = kargs.get('optimize_k', False)
    ###################################################
    
    # determine (sub-)optimal number of clusters
    
    if optimize_k and cluster_method.startswith(('kmean', 'k-mean')): 
        n_clusterx = []

        ### gap statistics 
        div(message='Determining best number of clusetrs (K) ...')
        rdim = kargs.get('reduce_dimension', False)
        step = 5
        min_n_clusters, max_n_clusters = kargs.get('min_n_clusters', 1), kargs.get('max_n_clusters', maxK)
        nrefs = 15

        print('params> n_clusters: (%d ~ %d) | step=%d, nrefs: %d' % (min_n_clusters, max_n_clusters, step, nrefs))
        n_clusters_gap = gap_stats.find_optimal_k(X, y=y, reduce_dimension=rdim, identifier=identifier, 
                                                        min_n_clusters=min_n_clusters, max_n_clusters=max_n_clusters,
                                                            step=step, output_dir=output_dir)

        # [log] status> best n_clusters (by gap statistic): 62 vs requested 50
        print('status> best K (gap statistic): %d in range (%d, %d) | requested: %d' % \
            (n_clusters_gap, min_n_clusters, max_n_clusters, n_clusters))
        n_clusterx.append(n_clusters_gap)
        n_clusters = int(math.ceil(np.mean(n_clusterx)))
        # print('status> best k (after averaging): %d' % n_clusters)
        
        ### Silhouette scores 
        range_n_clusters = kargs.get('range_n_clusters', None)
        if range_n_clusters is None: 
            range_n_clusters = range(max(n_clusters_gap-10, 1), min(n_clusters_gap+10, X.shape[0]))
            for candidate_k in (2, 5, 10, 20, 50, 100):
                if not candidate_k in range_n_clusters and candidate_k < maxK: 
                    bisect.insort_left(range_n_clusters, candidate_k)
            kargs['range_n_clusters'] = range_n_clusters

        n_clusters_silh = run_silhouette_analysis(X, y, **kargs) # args: range_n_clusters > default [2, 3, 4, 5, 6, 10, 15, 20]
        print('status> best n_clusters (by silhouette test): %d from range(%s)' % (n_clusters_silh, range_n_clusters))
        # n_clusters = n_clusters_silh
        n_clusterx.append(n_clusters_silh)
        
        ### final K 
        # n_clusters = int(math.ceil(np.mean(n_clusterx)))
        n_clusters = n_clusters_gap
        k_optimized = True

    if cluster_method in ('kmeans', 'k-means', ):  
        if k_optimized: print('kmeans> using optimized K: %d' % n_clusters)
        model = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        model.fit(X)
    elif cluster_method in ('minibatch', 'minibatchkmeans'):
        model = MiniBatchKMeans(n_clusters=n_clusters)  # init='k-means++', n_init=3 * batch_size, batch_size=100
        model.fit(X)
    elif cluster_method.startswith('spec'):
        model = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity="nearest_neighbors") 
        model.fit(X)
    elif cluster_method.startswith('agg'):   # doesn't require n_clusters
        knn_graph = kneighbors_graph(X, 30, include_self=False)  # n_neighbors: Number of neighbors for each sample.

        # [params] AgglomerativeClustering
        # connectivity: Connectivity matrix. Defines for each sample the neighboring samples following a given structure of the data. 
        #               This can be a connectivity matrix itself or a callable that transforms the data into a connectivity matrix
        # linkage:  The linkage criterion determines which distance to use between sets of observation. 
        #           The algorithm will merge the pairs of cluster that minimize this criterion.
        #           average, complete, ward (which implies the affinity is euclidean)
        # affinity: Metric used to compute the linkage
        #           euclidean, l1, l2, manhattan, cosine, or precomputed
        #           If linkage is ward, only euclidean is accepted.
        connectivity = knn_graph # or None
        linkage = kargs.get('linkage', 'average')
        model = AgglomerativeClustering(linkage=linkage,  
                                            connectivity=connectivity,
                                            n_clusters=n_clusters)
        model.fit(X)
    elif cluster_method.startswith('aff'): # affinity propogation
        damping = kargs.get('damping', 0.9)
        preference = kargs.get('preference', -50)

        # expensive, need subsampling
        model = AffinityPropagation(damping=damping, preference=preference) 
        model.fit(X)
        
        cluster_centers_indices = model.cluster_centers_indices_
        n_clusters_est = len(cluster_centers_indices)
        print('affinityprop> method: %s (damping: %f, preference: %f) > est. n_clusters: %d' % (cluster_method, damping, preference, n_clusters)) 

    elif cluster_method.startswith('db'): # DBSCAN: density-based 
        # first estimate eps 
        n_sample_max = 500
        metric = kargs.get('metric', distance.cosine) # 'euclidean'

        # [note] 
        # eps : float, optional
        #       The maximum distance between two samples for them to be considered as in the same neighborhood.
        # min_samples : int, optional
        #       The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. 
        #       This includes the point itself.
        
        eps = kargs.get('eps', None)
        if eps is None:      
            X_subset = X[np.random.choice(X.shape[0], n_sample_max, replace=False)] if X.shape[0] > n_sample_max else X

            # pair-wise distances
            dmat = distance.cdist(X_subset, X_subset, metric)
            off_diag = ~np.eye(dmat.shape[0],dtype=bool)  # don't include distances to selves
            dx = dmat[off_diag]
            sim_median = np.median(dx)
            sim_min = np.min(dx)
            eps = (sim_median+sim_min)/2.

        print('(cluster_analysis) Method: %s | eps: %f, metric: %s' % (cluster_method, eps, metric))
        model = DBSCAN(eps=eps, min_samples=10, metric=metric) 
        model.fit(X)

        core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        core_samples_mask[model.core_sample_indices_] = True

        y_pred = model.labels_
        n_clusters_est = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    else: 
        raise NotImplementedError("Cluster method %s is not yet supported" % cluster_method )

    return (model, n_clusters_est)

def run_silhouette_analysis(X, y, **kargs):
    from sklearn.metrics import silhouette_samples, silhouette_score
    # import matplotlib.cm as cm
    tFoundManifoldMethod = False 
    try: 
        import learn_manifold
        tFoundManifoldMethod = True
    except: 
        pass 

    # [params] input 
    assert X is not None and X.shape[0] > 1    
    N = X.shape[0] 
    n_clusters_max = max(2, N/2)

    range_n_clusters = kargs.get('range_n_clusters', range(2, n_clusters_max, 5))
    n_clusters_min, n_clusters_max = min(range_n_clusters), max(range_n_clusters)
    identifier = kargs.get('identifier', 'nCm%d_M%d' % (n_clusters_min, n_clusters_max))
    dim0 = X.shape[1]
    if kargs.get('reduce_dimension', False) and tFoundManifoldMethod: # dimensionality reduction prior to gap statistical analysis
        X = learn_manifold.tsne(X, identifier=identifier)  # use t-SNE by default
        print('run_silhouette_analysis> dim of X from %d to %d' % (dim0, X.shape[1]))

    # [params]
    # range_n_clusters = kargs.get('range_n_clusters', [2, 3, 4, 5, 6, 10, 15, 20])
    n_clusters_requested = kargs.get('n_clusters', None)
    if n_clusters_requested is not None: 
        if not n_clusters_requested in range_n_clusters: 
            range_n_clusters.append(n_clusters_requested)
    print('param> input n_clusters (requested): %s > range_n_clusters: %s' % (n_clusters_requested, range_n_clusters))

    # identifier 
    identifier = kargs.get('identifier', 'nR%s-%s' % (min(range_n_clusters), max(range_n_clusters)))
    output_dir = kargs.get('output_dir', os.path.join(os.getcwd(), 'plot'))
    if not os.path.exists(output_dir): os.makedirs(output_dir) # base directory

    ranked_scores = []
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        plt.clf()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        y_pred = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, y_pred)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        ranked_scores.append((n_clusters, silhouette_avg))

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, y_pred)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[y_pred == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(y_pred.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                     marker='o', c="white", alpha=1, s=200)
   
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        # plt.show()
        graph_ext = 'tif'
        fpath = os.path.join(output_dir, 'silhouette_test-%s-nC%s.%s' % (identifier, n_clusters, graph_ext))
        print('output> saving silhouette test result to %s' % fpath)
        plt.savefig(fpath)
    ### end range of n_clusters 

    ranked_scores = sorted(ranked_scores, key=lambda x: abs(x[1]), reverse=False) # reverse=False => ascending 
    print('output> ranked scores (n_clusters vs average score):\n%s\n' % ranked_scores)

    return ranked_scores[0][0]

def spectral_cluster(X, n_clusters, **kargs): 
    """

    Params
        affinity: a string or a function 

    **kargs 
        eigen_solver

    """
    affinity = kargs.get('affinity', None)  # None, if X is a precomputed affinity matrix, or it can be a string or a function

    # from sklearn.cluster import SpectralClustering
    if isinstance(affinity, str): 
        model = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity=affinity) 
        model.fit(X)
    elif hasattr(affinity, '__call__'):  
        A = affinity(X)  

        # SpectralClustering(n_clusters=n,affinity='precomputed').fit_predict(dA)
        model = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity='precomputed') # fit_predict(affinity)
        model.fit(A)
    else:  # affinity matrix has been precomputed 
        model = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity='precomputed') # fit_predict(affinity)
        model.fit(X)

    return model.labels_

def demo_clustering(output_dir=None):
    from utils import saveFig, highlight
    from cluster_utils import getAffinityMatrix, eigenDecomposition

    # import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    import hdbscan  # Hierarchical DBSCAN (pip install hdbscan)

    # Parameters
    ###################################
    if output_dir is None: output_dir = os.path.join(os.getcwd(), 'output')

    ###################################

    plt.figure(figsize=(12, 12))

    highlight("Generating cluster data ...")
    n_clusters = 7
    X, y_true = make_blobs(n_samples=300, centers=n_clusters,
                           cluster_std=.80, random_state=0)
    plt.title(f'Ground truth simulated data : {n_clusters} clusters')
    plt.scatter(X[:, 0], X[:, 1], s=50, c = y_true);

    fpath = os.path.join(output_dir, f'scatter_plot_{n_clusters}_clusters')
    # saveFig(plt, fpath, ext='tif', dpi=250, message='', verbose=True)
    plt.savefig(fpath, dpi=300)

    # Run Density-based clustering 
    highlight("Running Density-based clustering e.g. HDBSCAN")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    y_pred = clusterer.fit_predict(X)
    n_clusters_est = len(np.unique(y_pred))
    print(f"(demo) Estimated number of clusters by HDBSCAN: {n_clusters_est} ~? 'True' #: {n_clusters}")

    highlight(f"Running HDBSCAN with n={len(np.unique(y_true))} unique true labels") 
    cluster_analysis(X, y_true, model=clusterer, cluster_method='hdbscan') 

    print('\n\n')
    # Run Graph-based clustering
    ###################################
    plt.clf()

    highlight("Running Graph-based clustering e.g. Spectral Clustering ...")
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0).fit(X)
    
    y_pred = clustering.labels_
    plt.figure(figsize=(14,6))
    plt.subplot(121)
    plt.title(f'Spectral clustering results ')
    plt.scatter(X[:, 0], X[:, 1], s=50, c = y_pred);

    plt.subplot(122)
    plt.title(f'Ground truth clustering ')
    plt.scatter(X[:, 0], X[:, 1], s=50, c = y_true);

    fpath = os.path.join(output_dir, f'clusters_vs_groundtruth')
    # saveFig(plt, fpath, ext='tif', dpi=250, message='', verbose=True)
    plt.savefig(fpath, dpi=300)

    print('\n')
    highlight("Compute affinity matrix and estimate (optimal) number of parameters")
    affinity_matrix = getAffinityMatrix(X, k = 10)
    suggested_k, _,  _ = eigenDecomposition(affinity_matrix)
    print(f'Optimal number of clusters {suggested_k}')

    n_clusters = min(suggested_k)
    clusterer = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0)
    y_pred = clusterer.fit(X)

    highlight(f"Running cluster analysis with n={len(np.unique(y_true))} unique true labels") 
    cluster_analysis(X, y_true, model=clusterer,
        n_clusters=n_clusters, cluster_method='spectral') 

    return

def test(**kargs): 

    ## cluster analysis 
    # demo_cluster_analysis() 
    
    # Partition-based and graph-based clustering
    demo_clustering()


    return 


if __name__ == "__main__": 
    test() 


