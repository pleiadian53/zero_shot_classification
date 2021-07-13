# encoding: utf-8

import os
import numpy as np
import collections

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from evaluate import calculate_metrics
from utils import highlight

import seaborn as sns
import matplotlib
# matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt

# select plotting style 
plt.style.use('seaborn')  # values: {'seaborn', 'ggplot', }

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy')>0.95:
            print("\nReached 95.00% accuracy so cancelling training!")
            self.model.stop_training = True

def create_baseline(n_features, n_hidden=1024, n_hidden2=6):
    # create model
    if n_hidden is None: n_hidden = int(n_features/10)

    model = tf.keras.models.Sequential()
    model.add(Dense(n_hidden, input_dim=n_features, activation='relu')) 
    # model.add(Dropout(0.2)),
    # model.add(Dense( n_hidden2, activation='relu')) # kernel_regularizer=regularizers.l2(0.01)
    model.add(Dropout(0.2)),
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                        metrics=['accuracy',  tf.keras.metrics.AUC()]) 

    return model 

def create_baseline2(n_features, n_hidden=512, n_hidden2=256):
    
    input_layer = Input(shape=(n_features,))
    hidden1 = Dense(n_hidden, kernel_regularizer=regularizers.l2(0.01), activation='relu')(input_layer)
    # hidden1 = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden1)
    # hidden2 = Dense(n_hidden2, activation='relu')(hidden1)
    output = Dense(1, activation='sigmoid')(hidden1)

    model = Model(inputs=input_layer, outputs=output)

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                        metrics=['accuracy',  tf.keras.metrics.AUC()])  

    return model 

def plot_graphs(history, string):
    plt.clf()
    print(f"(plot_graph) Plotting {string} ...")
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
    return

def classify_document_concept_vectors(num_epochs=10, n_trials=1, n_samples=None, random_state=53): 
    # from utils import highlight
    from data_pipeline import load_document_concept_vectors
    # from evaluate import calculate_metrics

    callbacks = myCallback()

    # Get the training split and the remaining split that is associated with concepts/keywords ...
    # ... not observed in the training set 
    X_train, X_test, y_train, y_test = load_document_concept_vectors(data_index=0)
    N, n_features = X_train.shape
    Nt = X_test.shape[0]
    # ... Note that the training split is fixed (somewhat time-consuming to create)

    # Sample a subset of the training data
    if n_samples is not None and n_samples < N: 
        idx = np.random.choice(N, n_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    
    label_counts = collections.Counter(y_train)

    highlight(f"Training set size: {N}, size(dev+test): {Nt}, n_features: {n_features}")
    print(f"... n(+): {label_counts[1]} ~? n(-): {label_counts[0]}")

    useDevSet = False
    for i in range(0, n_trials): 

        # Further create a validation set and a test set
        if useDevSet:  
            X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state)
        else: 
            X_valid, y_valid = X_test, y_test

        print(f"[{i+1}] size(train): {X_train.shape[0]}, size(valid): {len(X_valid)}, size(test): {len(X_test)}")
        
        model = create_baseline2(n_features=300*2, n_hidden=100) 
        model.summary()

        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
            
        history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_valid, y_valid), 
                               batch_size=64,
                               callbacks=[callbacks, ], 
                               verbose=1) # lr_schedule

        # plot_graphs(history, 'accuracy')
        # plot_graphs(history, 'auc')
        # plot_graphs(history, 'loss')
        analyze_performance(history)

        print(f"[status] Trial #{i+1} training complete. Predicting new data ...")

        # Ideally, we want to retrain the model by combining the data from the training split and validation split
        # This also needs to ensure that the new test split is associated with the concepts NOT observed in train+valid
        # However, this step is skipped in this demo

        if useDevSet: 
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred, p_th=0.5, phase='test')
            print(f"... On unseen data > Accuracy: {metrics['accuracy']}, AUC: {metrics['AUROC']}")

    return model


def analyze_performance(history, output_dir=None, show_plot=True): 

    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    
    auc=history.history['auc']
    val_auc=history.history['val_auc']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs=range(len(acc)) # Get number of epochs

    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.clf()
    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Validation Accuracy"])

    if show_plot: plt.show()
    if output_dir is None: output_dir = os.path.join(os.getcwd(), 'output')
    fpath = os.path.join(output_dir, f'training-vs-validation-accuracy')
    plt.savefig(fpath, dpi=300)

    #------------------------------------------------
    # Plot training and validation AUC per epoch
    #------------------------------------------------
    plt.clf()
    plt.plot(epochs, auc, 'r')
    plt.plot(epochs, val_auc, 'b')
    plt.title('Training and validation AUC')
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.legend(["AUC", "Validation AUC"])

    if show_plot: plt.show()
    if output_dir is None: output_dir = os.path.join(os.getcwd(), 'output')
    fpath = os.path.join(output_dir, f'training-vs-validation-AUC')
    plt.savefig(fpath, dpi=300)

    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.clf()
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation Loss"])

    if show_plot: plt.show()
    fpath = os.path.join(output_dir, f'training-vs-validation-loss')
    plt.savefig(fpath, dpi=300)

    return

def cluster_document_vectors(input_file='news_data.csv', output_dir=None, **kargs): 
    """

    Reference 
    ---------
    1. HDBSCAN: https://github.com/scikit-learn-contrib/hdbscan
    """
    def subsample(X, y=None, n=2000): 
        selected = np.random.choice(X.shape[0], min(n, X.shape[0]), replace=False)
        if y is not None and len(y) > 0: 
            assert len(y) == X.shape[0]
            return X[selected], y[selected]
        return X[selected], None

    from pandas import DataFrame
    from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, SpectralClustering
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from data_pipeline import load_data, save_data, load_document_vectors
    from utils import saveFig, highlight
    from cluster_utils import getAffinityMatrix, eigenDecomposition
    import learn_manifold
    from cluster import cluster_analysis
    import hdbscan # pip install hdbscan

    # Parameters
    ###################################
    if output_dir is None: output_dir = os.path.join(os.getcwd(), 'output')

    n_clusters = kargs.get('n_clusters', 10)
    n_samples = kargs.get('n_samples', None)
    n_samples_plot = kargs.get('n_samples_plot', 100)

    min_cluster_size = 20  # used for density-based clustering method
    method = kargs.get('method', 'default')   # 'density' to use HDBSCAN
    ###################################

    df = load_data(input_file=input_file, subset=False, verbose=1)
    N = df.shape[0]
    X, y_true = load_document_vectors(data_index=0)  
    assert X.shape[0] == N
    print(f"> dim(X): {X.shape}")

    if n_samples is not None: 
        X, y_true = subsample(X, y_true, n=n_samples)
        N = X.shape[0]
        highlight(f"Subsampled a subset of the data (n={N})")

    # Run clustering
    ######################################################################  
    # plt.figure(figsize=(12, 12))
    # plt.clf()

    if method.startswith('den'):  # density-based method, a good option would hierarchical DBSCAN
        # Xp = PCA(n_components=20).fit_transform(X)
        highlight(f"Running HDBSCAN on n={X.shape[0]} document vectors of dimension {X.shape[1]} ...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        y_pred = clusterer.fit_predict(X)
        assert len(y_pred) == N

        n_clusters = len(np.unique(y_pred))
        highlight(f"Estimated number of clusters by HDBSCAN: {n_clusters} given mininum cluster size: {min_cluster_size}")
        cluster_analysis(Xp, y=None, model=clustering, 
            n_clusters=n_clusters, cluster_method='k-means') 
    else: 
        highlight(f"Running PCA followed by K-Means on n={X.shape[0]} document vectors of dimension {X.shape[1]} ...")
        Xp = PCA(n_components=10).fit_transform(X)

        clustering = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10).fit(Xp)
        # clustering = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, batch_size=1000, n_init=10).fit(Xp)
        y_pred = clustering.labels_
        assert len(y_pred) == N

        cluster_analysis(Xp, y=None, model=clustering, 
            n_clusters=n_clusters, cluster_method='k-means') 

    # Save clustering results
    ###################################################################### 
    if len(y_pred) == df.shape[0]: # if no subsampling was applied (and the ordering of the data is consistent)
        df['cluster_id'] = y_pred
        print(f"(demo) Saving cluster IDs to {input_file}")
        save_data(df, output_file="news_data.csv")

    # Cluster analysis only on a subset of the data
    ###################################################################### 
    plt.clf()
    n_small_samples = 2000

    X_subset, y_subset = subsample(X, y_pred, n=n_small_samples)
    var_cols = [ 'var'+str(i) for i in range(X_subset.shape[1]) ]
    df_subset = DataFrame(X_subset, columns=var_cols) 
    df_subset['y'] = y_subset

    highlight(f"Running T-SNE on a subset of the data (n={n_small_samples}) ...")
    X_embedded = TSNE(n_components=2, perplexity=45, verbose=1).fit_transform(X_subset)
    
    plt.figure(figsize=(16,10))
    plt.title(f'Running T-SNE on n={n_small_samples} Document-Concept Vectors')
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=50, c = y_subset);
    df_subset['Dimension-1'] = X_embedded[:,0]
    df_subset['Dimension-2'] = X_embedded[:,1]
    sns.scatterplot(
        x='Dimension-1', y='Dimension-2',
        hue='y',
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )

    fpath = os.path.join(output_dir, f'document_clusters-tsne')
    plt.show()
    saveFig(plt, fpath, ext='tif', dpi=250, message='', verbose=True)

    # Compute affinity matrix and estimate (optimal) number of parameters
    ###################################################################### 
    plt.clf()
    highlight(f"Estimate # of clusters via eigengap and run spectral clustering accordingly (n={n_small_samples}) ...")

    affinity_matrix = getAffinityMatrix(X_subset, k = 10)
    suggested_k, _,  _ = eigenDecomposition(affinity_matrix)
    print(f'... Optimal number of clusters {suggested_k}')

    n_clusters = min(max(suggested_k), 10)
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0).fit(X_subset)

    # if len(y_true) > 0: highlight(f"[demo] Run cluster analysis with n={len(np.unique(y_true))} unique true labels") 
    cluster_analysis(X_subset, y=None, model=clustering, 
        n_clusters=n_clusters, cluster_method='spectral') 

    return 

def test_document_concept_vectors(model=None, n_trials=1, n_samples=None, random_state=53, verbose=0): 
    """
    Examine the preditability of the document-concept vector via classical ML methods 
    such as logistic regression, random forest among other non-NN-based algorithms. 
    """
    from data_pipeline import load_document_concept_vectors
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.ensemble import RandomForestClassifier

    # Get the training split and the test split associated with concepts/keywords ...
    # ... not observed in the training set
    X_train, X_test, y_train, y_test = load_document_concept_vectors(data_index=0)

    N, n_features = X_train.shape
    N_test = X_test.shape[0]
    # ... Note that the training split is fixed (time-consuming to create)

    # Sample a subset of the training data
    if n_samples is not None and n_samples < N: 
        highlight(f"[test] Sampling n={n_samples} data points ...")
        idx = np.random.choice(N, n_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    
    label_counts = collections.Counter(y_train)

    highlight(f"Training set size: {N}, size(test): {N_test}, n_features: {n_features}")
    print(f"... n(+): {label_counts[1]} ~? n(-): {label_counts[0]}")

    if not model: 
        # model = LogisticRegression(penalty='l2', solver='sag', random_state=random_state, verbose=verbose)
        # model = SGDClassifier(loss='log', penalty='l2', max_iter=1000, verbose=verbose) # tol=1e-3
        model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=10, min_samples_leaf=10, verbose=verbose)
        print(f"... Using classifier: {model.__class__.__name__}")  # hasattr(model, '__class__')

    for i in range(0, n_trials): 

        print(f"[{i+1}] size(train): {X_train.shape[0]}, size(test): {X_test.shape[0]}")
            
        model.fit(X_train, y_train)

        print(f"[status] Trial #{i+1} training complete. Predicting new data ...")

        y_pred = model.predict_proba(X_test)[:,1]
        print(f">>> y_pred dim: {y_pred.shape}, y_pred:\n{y_pred[:10]}\n")
        metrics = calculate_metrics(y_test, y_pred, p_th=0.5, phase='test')
        print(f"... On unseen data > Accuracy: {metrics['accuracy']}, AUC: {metrics['AUROC']}")


    return

def test_dnn(n_trials=1, use_synthetic=True): 
    """ 
    Test the NN-based classifier on real datasets. 
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.datasets import make_classification
    # from evaluate import calculate_metrics

    #################################
    callbacks = myCallback()
    num_epochs = 100
    #################################

    if use_synthetic: 
        # Generate synthetic dataset 
        n_features = 600
        highlight("Generate synthetic dataset ...")
        X, y = make_classification(n_samples=100000, n_features=n_features, n_redundant=0, n_informative=372,
                               random_state=53, n_clusters_per_class=3, n_classes=2)
        rng = np.random.RandomState(2)
        X += 2 * rng.uniform(size=X.shape)
    else: 
        X, y = load_breast_cancer(return_X_y=True)
        n_features = X.shape[1]

    for i in range(0, n_trials): 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=53)

        print(f"[{i+1}] size(train): {X_train.shape[0]}, size(test): {X_test.shape[0]} | n_features: {n_features}")
        
        model = create_baseline2(n_features=n_features, n_hidden=10)
        model.summary()
        # lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
            
        history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), 
                               callbacks=[callbacks, ], 
                               verbose=1) # lr_schedule

        # plot_graphs(history, 'accuracy')
        # plot_graphs(history, 'auc')
        # plot_graphs(history, 'loss')
        analyze_performance(history)

        print(f"[status] Trial #{i+1} training complete. Predicting new data ...")


    return

def run_modeling_pipeline(input_file='news_data.csv', test_mode=False): 

    # Compute document vectors and document-concept vectors
 
    n_samples = None
    n_samples = None if not test_mode else 2000  # or any desirable sample size

    # Cluster analysis 
    cluster_document_vectors(input_file=input_file, n_samples=n_samples)

    # Document-concept association classification
    classify_document_concept_vectors(num_epochs=100, n_samples=n_samples) 

    # Test the NN model on other datasets and test classical classifers on the document-concept-vector dataset
    if test_mode:  
        # test_dnn()

        # Test the predictability of the document-concept vector with classifical ML methods
        test_document_concept_vectors(n_samples=n_samples, verbose=True)

    return

def test(): 

    # Run pipeline 
    run_modeling_pipeline(test_mode=False)

    # test_dnn()

    return

if __name__ == "__main__": 
    test()