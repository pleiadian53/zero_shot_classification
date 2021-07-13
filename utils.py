
import os, sys
from pandas import DataFrame, Series
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator # used to force yticks to assume integer values
plt.style.use('ggplot')

# -- Word2Vec -- 
def get_vectors(embeddings, words):
    """
    Input:
        embeddings: a word 
        fr_embeddings:
        words: a list of words
    Output: 
        X: a matrix where the rows are the embeddings corresponding to the rows on the list
        
    """
    m = len(words)
    X = np.zeros((1, 300))
    for word in words:
        english = word
        eng_emb = embeddings[english]
        X = np.row_stack((X, eng_emb))
    X = X[1:,:]
    return X


# -- Search -- 
#######################################################
def search(file_patterns=None, basedir=None): 
    """

    Memo
    ----
    1. graphic patterns 
       ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']

    """
    import fnmatch

    if file_patterns is None: 
        file_patterns = ['*.dat', '*.csv', ]
    if basedir is None: 
        basedir = os.getcwd()

    matches = []

    for root, dirnames, filenames in os.walk(basedir):
        for extensions in file_patterns:
            for filename in fnmatch.filter(filenames, extensions):
                matches.append(os.path.join(root, filename))
    return matches


# -- Plotting -- 
#######################################################
def plot_roc(cv_data, **kargs):
    """
    
    Params
    ------
    cv_data: a list of (y_true, y_score) obtained from a completed CV process
    """
    from scipy import interp
    from sklearn.metrics import roc_curve, auc

    method = kargs.get("method", "test")
    prefix_id = kargs.get('prefix', '')
    show_plot = kargs.get("show", False)
    verbose = kargs.get("verbose", True)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    n_fold = len(cv_data)
    if not n_fold: 
        print('(plot_roc) No CV data. Aborting ...')
        return

    plt.clf()
    for i, (y_true, y_score) in enumerate(cv_data):

        fold_num = i+1
        if (n_fold > 5 and fold_num % 2 == 0) or n_fold <= 5: 
            # probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_true, y_score) # roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
     
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                        label='ROC fold %d (AUC = %0.2f)' % (fold_num, roc_auc))

    ### plotting
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if prefix_id: 
        plt.title('ROC for {}: method={}, nfold={})'.format(prefix_id, method, n_fold)) 
    else: 
        plt.title('ROC for {}, nfold={})'.format(method, n_fold))
    plt.legend(loc="lower right")
    
    filename = kargs.get('filename', 'roc')
    if prefix_id: 
        filename = f"{prefix_id}-roc"
    output_dir = kargs.get("output_dir", "plot")
    output_path = os.path.join(output_dir, filename)  # example path: System.analysisPath
    if verbose: 
        print(f"[plot] Saving ROC curve to:\n{output_path}\n")
    saveFig(plt, output_path, dpi=300, message="[output] ROC curve", verbose=verbose)

    if show_plot: 
        plt.show()

    return

def saveFig(plt, fpath, ext='tif', dpi=300, message='', verbose=True):
    """
    Save plots. 

    Params
    ------
    fpath: output path of the plot
    message: stdout message for debugging/test purposes

    Memo
    ----
    1. supported graphing format: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff 

    """
    import os
    outputdir, fname = os.path.dirname(fpath), os.path.basename(fpath) 

    # [todo] abstraction
    supported_formats = ['eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff', ]  

    # supported graphing format: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    if not outputdir: outputdir = os.getcwd() # sys_config.read('DataExpRoot') # ./bulk_training/data-learner)
    # assert os.path.exists(outputdir), "Invalid output path: %s" % outputdir
    if not os.path.exists(outputdir):
        if verbose: print("(savefig) Creating directory: {}".format(outputdir))
        os.mkdir(outputdir)

    ext_plot = ext  # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    if not fname: fname = 'generic-test.%s' % ext_plot
    fbase, fext = os.path.splitext(fname)
    # if fext[1:] in supported_formats, "Unsupported graphic format: %s" % fname
    if not fext: fname = fbase + '.tif' 

    fpath = os.path.join(outputdir, fname)
    if verbose: print('(saveFig) Saving plot to:\n%s\n... description: %s' % (fpath, 'n/a' if not message else message))
    
    # pylab leaves a generous, often undesirable, whitespace around the image. Remove it by setting bbox_inches to tight
    plt.savefig(fpath, bbox_inches='tight', dpi=dpi)   
    return

# -- Formatting -- 
#######################################################
def indent(message, nfill=6, char=' ', mode='r'): 
    if mode.startswith('r'): # left padding 
        return message.rjust(len(message)+nfill, char)
    return message.ljust(len(message)+nfill, char)

def div(message=None, symbol='=', prefix=None, n=80, adaptive=False, border=0, offset=5, stdout=True): 
    output = ''
    if border is not None: output = '\n' * border
    # output += symbol * n + '\n'
    if isinstance(message, str) and len(message) > 0:
        if prefix: 
            line = '%s: %s\n' % (prefix, message)
        else: 
            line = '%s\n' % message
        if adaptive: n = len(line)+offset 

        output += symbol * n + '\n' + line + symbol * n
    elif message is not None: 
        # message is an unknown object of some class
        if prefix: 
            line = '%s: %s\n' % (prefix, str(message))
        else: 
            line = '%s\n' % str(message)
        if adaptive: n = len(line)+offset 
        output += symbol * n + '\n' + line + symbol * n
    else: 
        output += symbol * n
        
    if border is not None: 
        output += '\n' * border
    if stdout: print(output)
    return output
### alias 
highlight = div


# -- Data Structure -- 
#######################################################
def summarize_dict(d, topn=15, sort_=True, prefix=''): 
    if topn != 0 or sort_: 
        import operator
        d = sorted(d.items(), key=operator.itemgetter(1))
    for k, v in d[:topn]: 
        print(f"{prefix}[{k}] -> {v}")
    return

def least_common(counts, n=None):
    counter = collections.Counter(counts)
    if n is None:
        return sorted(counter.items(), key=itemgetter(1), reverse=False)
    return heapq.nsmallest(n, counter.items(), key=itemgetter(1))

def dictToList(adict):
    lists = []
    for k, v in nested_dict_iter(adict): 
        alist = []
        if not hasattr(k, '__iter__'): k = [k, ]
        if not hasattr(v, '__iter__'): v = [v, ]
        alist.extend(k)
        alist.extend(v)
        lists.append(alist)
    return lists

def nested_dict_iter(nested):
    import collections

    for key, value in nested.iteritems():
        if isinstance(value, collections.Mapping):
            for inner_key, inner_value in nested_dict_iter(value):
                yield inner_key, inner_value
        else:
            yield key, value

def dictSize(adict): # yeah, size matters  
    return len(list(nested_dict_iter(adict)))
def size_dict(adict): 
    """

    Note
    ----
    1. size_hashtable()
    """
    return len(list(nested_dict_iter(adict)))


def partition(lst, n):
    """
    Partition a list into almost equal intervals as much as possible. 
    """
    q, r = divmod(len(lst), n)
    indices = [q*i + min(i, r) for i in xrange(n+1)]
    return [lst[indices[i]:indices[i+1]] for i in xrange(n)]

def divide_interval(total, n_parts):
    pl = [0] * n_parts
    for i in range(n_parts): 
        pl[i] = total // n_parts    # integer division

    # divide up the remainder
    r = total % n_parts
    for j in range(r): 
        pl[j] += 1

    return pl 