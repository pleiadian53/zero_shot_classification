import numpy as np
import os
from random import randrange
from pandas import DataFrame, Series

import sklearn.metrics
from sklearn.metrics import plot_roc_curve, precision_recall_curve, f1_score, roc_auc_score, auc, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold # Import train_test_split function

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
# from sklearn import svm, datasets

import matplotlib
# matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
matplotlib.use('TkAgg')  # using this will allow plt.show() later on
from matplotlib import pyplot as plt
from utils import saveFig, plot_roc


def eval_performance(X, y, model, cv=5, random_state=53, **kargs):
    """

    Memo
    ----
    1. https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

    """
    # from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
    import time
    
    #### parameters ####
    verbose = kargs.get('verbose', 0)
    algo_name = kargs.get('method', 'LG')
    output_dir = kargs.get('output_dir', 'plot')
    plot_prefix = kargs.get('prefix', '')
    save_plot = kargs.get('save', True)
    show_plot = kargs.get('show', False)
    # plot_ext = kargs.get('ext', 'tif')
    ####################

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    if len(y.shape) == 1: 
        y = y.reshape((y.shape[0], 1))

    cv_scores =[]
    cv_data = []
    predictions = {i: {} for i in range(cv)}
    fold_number_act = np.random.randint(0, cv)
    for i, (train, test) in enumerate(kf.split(X,y)):
        fold_number = i
        if verbose: 
            print('[eval] {} of KFold {}'.format(fold_number, kf.n_splits))

        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        
        #model
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:,1]

        #evaluation 
        cv_data.append((y_test, y_pred))
        score, p_th = fmax_score_threshold(y_test, y_pred)
        score_auc = roc_auc_score(y_test, y_pred)

        # other metrics
        y_pred_label = model.predict(X_test) # to_label_prediction(y_pred, p_th=0.5) #
        score_f1 = f1_score(y_test, y_pred_label)

        predictions[i]['scores'] = calculate_metrics(y_test, y_pred, 
            plot=save_plot, show=show_plot and i==fold_number_act,
            method=algo_name, index=fold_number, phase='test', prefix=plot_prefix) 
        # ... add a prefix to plot name to distinguish between different training data sets

        if verbose: 
            print('> Fmax: {} p_th: {} | F1: {}, AUC: {}'.format(score, p_th, score_f1, score_auc))
        
        cv_scores.append(score)    

    # [note]
    # The ROC curve within each CV fold can be combined while still maintaining visual clarity
    # However, it's not the case for the PR curve, and therefore, I've decided to 
    # plot the PR curve seperately for each CV fold.
    if save_plot: 
        plot_roc(cv_data, output_dir=output_dir, method=algo_name, prefix=plot_prefix, show=show_plot)

    return cv_scores, predictions

def to_label_prediction(y_score, p_th=0.5):
    # turn probabilities into label prediction given threshold at 'p_th'
    yhat = np.zeros(len(y_score))
    for i, yp in enumerate(y_score): 
        if yp >= p_th: 
            yhat[i] = 1
    return yhat

def calculate_accuracy(y_true, y_hat):
    """
    Calculate accuracy percentage  
    """
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_hat[i]:
            correct += 1
    return correct / float(len(y_true))

def eval_AUPRC(y_true, y_score, method='LG', plot=True, **kargs): 
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)

    show_plot = kargs.get('show', False)
    phase = kargs.get("phase", 'test') 
    prefix_id = kargs.get('prefix', '')

    if plot:  # plot the precision-recall curves
        plt.clf()
        fold_number = kargs.get('index', 0)

        y_true = np.array(y_true)
        no_skill = len(y_true[y_true==1]) / len(y_true)

        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(recall, precision, marker='.', label=method)
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        if prefix_id: 
            plt.title('PRC for {}: method={}, nfold={})'.format(prefix_id, method, fold_number)) 
        else: 
            plt.title('PRC for {}, nfold={})'.format(method, fold_number))

        # show the legend
        plt.legend()
        
        if prefix_id: 
            prefix = f'{prefix_id}-precision_recall_curve-train' % prefix_id if phase.startswith('tr') else f'{prefix_id}-precision_recall_curve' 
        else:
            prefix = 'precision_recall_curve-train' % prefix_id if phase.startswith('tr') else 'precision_recall_curve'
        filename = kargs.get('filename', '{}-{}'.format(prefix, fold_number) if fold_number > 0 else prefix)
        output_dir = kargs.get("output_dir", "plot")
        output_path = os.path.join(output_dir, filename)  # example path: System.analysisPath
        saveFig(plt, output_path, ext='tif', dpi=300, message='[output] precision recall curve', verbose=True)
        
        # show the plot
        if show_plot: 
            plt.show()
    return auprc

def eval_AUROC(y_true, y_score, method='LG', plot=False, **kargs):
    auc = roc_auc_score(y_true, y_score)

    if plot: 
        pass # use utils_plot.plot_roc() instead

    # Note: ROC curve in each CV fold can be easily combined while still maintaining visual clarity, 
    #       and therefore, the plotting is deferred until all CV data are collected (see evaluate_algorithm())

    return auc

def calculate_metrics(y_true, y_score, p_th=0.5, **kargs):
    """
    Calculate performance metrics and plot related performance curves (e.g. precision-recall curve). 

    Params
    ------
    y_true: true labels 
    y_score: class conditional probabilities P(y=1|x)
    method: the name of the algorithm (for display only)
    phase: 'train' for the training phase or 'test' for the test phase 
           use this to make distinction between the plot associaetd with the training or test
           
           Most use cases only generate performance plot on the test data and therefore 
           the file name for the plot does not have this keyword. 

           If, however, we wish to diagnose overfitting by comparing the performance gap 
           between the training phase and the test phase, the performance plot can be 
           generated accordingly but with the keyword 'train' added to the plots' file names. 


    """
    # optional plot params
    index = kargs.get("index", 0)
    plot = kargs.get("plot", True)
    method = kargs.get("method", 'LR')
    phase = kargs.get("phase", 'test') # 

    metrics = {}
    metrics['AUROC'] = eval_AUROC(y_true, y_score, index=index)
    metrics['AUPRC'] = eval_AUPRC(y_true, y_score, index=index, plot=plot, method=method, phase=phase, prefix=kargs.get('prefix', ''))

    y_hat = to_label_prediction(y_score, p_th=p_th)          
    # ret['f1'] = f1_score(y_true, y_hat)
    
    nTP = nTN = nFP = nFN = 0
    for i, y in enumerate(y_true): 
        if y == 1: 
            if y_hat[i] == 1: 
                nTP += 1
            else: 
                nFN += 1
        else: # y == 0 
            if y_hat[i] == 0: 
                nTN += 1 
            else: 
                nFP += 1
    metrics['precision'] = nTP/(nTP+nFP+0.0)
    metrics['recall'] = nTP/(nTP+nFN+0.0)
    metrics['accuracy'] = calculate_accuracy(y_true, y_hat)

    return metrics


def perturb(X, cols_x=[], cols_y=[], lower_bound=0, alpha=100.):
    def add_noise():
        min_nonnegative = np.min(X[np.where(X>lower_bound)])
        
        Eps = np.random.uniform(min_nonnegative/(alpha*10), min_nonnegative/alpha, X.shape)

        return X + Eps
    # from pandas import DataFrame

    if isinstance(X, DataFrame):
        from data_processor import toXY
        X, y, fset, lset = toXY(X, cols_x=cols_x, cols_y=cols_y, scaler=None, perturb=False)
        X = add_noise(X)
        dfX = DataFrame(X, columns=fset)
        dfY = DataFrame(y, columns=lset)
        return pd.concat([dfX, dfY], axis=1)

    X = add_noise()
    return X

def fmax_score(labels, predictions, beta = 1.0, pos_label = 1):
    """

    Reference 
    ---------
    Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
    Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.
    """
    # import sklearn.metrics

    precision, recall, threshold = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label)

    # the general formula for positive beta
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    # if beta == 1, then this is just f1 score, harmonic mean between precision and recall 
    # i = np.nanargmax(f1)

    # return (f1[i], threshold[i])
    return nanmax(f1)

def fmax_score_threshold(labels, predictions, beta = 1.0, pos_label = 1):
    """
    Return the fmax score and the probability threhold where the max of f1 (fmax) is reached
    """
    # import sklearn.metrics
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label)

    # the general formula for positive beta
    # ... if beta == 1, then this is just f1 score, harmonic mean between precision and recall 
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    i = np.nanargmax(f1)  # the position for which f1 is the max 
    th = threshold[i] if i < len(threshold) else 1.0    # len(threshold) == len(precision) -1 
    # assert f1[i] == nanmax(f1)
    return (f1[i], th)

def one_vs_all_encoding(y, target_label, codebook={'pos': 1, 'neg': 0}, col='ICD10', col_target='target'): 
    # inplace operation
    
    return np.where(y == target_label, codebook['pos'], codebook['neg'])

def get_sample_sizes(y, col='ICD10'): 
    import collections
    
    if isinstance(y, DataFrame): 
        sizes = collections.Counter( y[col].values )
    else: 
        # df is a numpy array or list
        if len(y.shape) == 2: 
            y = y.reshape( (y.shape[0],) )
        sizes = collections.Counter(y)
        
    return sizes # label/col -> sample size

def encode_labels(y, pos_label, neg_label=None, col_label='ICD10', codebook={}, verbose=1): 
    if not codebook: codebook = {'pos': 1, 'neg': 0, '+': 1, '-': 0}
        
    sizes = get_sample_sizes(y)
    n0 = sizes[pos_label]

    # col_target='target'
    y = one_vs_all_encoding(y, target_label=pos_label, codebook=codebook)

    sizes = get_sample_sizes(y)
    assert sizes[codebook['pos']] == n0

    print("(encode_labels) sample size: {}".format(sizes))
    
    return y

def demo_evaluate_performance():  
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from sklearn.datasets import load_iris
    from utils import summarize_dict

    n_folds = 5
    iris = load_iris() 
    X = iris.data[:, :2]
    y = iris.target
    y = encode_labels(y, pos_label=1)

    model = LogisticRegression(class_weight='balanced', penalty='l1', solver='saga')

    scores, predictions = eval_performance(X, y, model=model, cv=n_folds, random_state=53, prefix='iris')
    print("[demo] Fmax average: {}, std: {}".format(np.mean(scores), np.std(scores)))
    
    for i in range(n_folds):
        print(f"... Fold #[{i}]: ")
        summarize_dict(predictions[i], topn=15, sort_=True, prefix=' ' * 5)

    return scores

def test(): 

    # test evaluating performance
    demo_evaluate_performance()

    return

if __name__ == "__main__":
    test()