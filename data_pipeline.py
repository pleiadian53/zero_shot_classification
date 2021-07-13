import os
import ast
import collections
import pandas as pd 
from pandas import DataFrame, Series
import numpy as np
from utils import highlight
import nltk


# Define the 'tokenize' function that will include the steps previously seen
def tokenize(corpus):
    """
    Tokenization prior to word embedding. 
    """
    # data = re.sub(r'[,!?;-]+', '.', corpus)
    data = re.sub(r'[,!;-]+', '.', corpus)    # '?' 
    data = nltk.word_tokenize(data)  # tokenize string to words
    data = [ ch.lower() for ch in data
             if ch.isalpha()
             or ch == '.'
             or emoji.get_emoji_regexp().search(ch)
           ]
    return data

########################################################

def fraction_rows_missing(df, verbose=False, n=None):
    '''
    Return percent of rows with any missing
    data in the dataframe. 
    
    Input:
        df (dataframe): a pandas dataframe with potentially missing data
    Output:
        frac_missing (float): fraction of rows with missing data
    '''
    rows_null = df.isnull().any(axis=1)
    n_null = sum(rows_null)
    
    if verbose: # show rows with null values
        if n_null > 0: 
            print(f"(fraction_rows_missing) Rows with nulls (n={n_null}):\n{df[rows_null].head(n=n).to_string(index=False)}\n")
    
    return sum(rows_null)/df.shape[0]


def save_data(df, output_file="news_data.csv", output_dir=None, verbose=False): 
    if output_dir is None: output_dir = os.path.join(os.getcwd(), 'data')
    assert os.path.exists(output_dir)
    
    output_path = os.path.join(output_dir, output_file) 
    df.to_csv(output_path, sep=',', index=False, header=True)

    if verbose: 
        print(f"(save_data) Dim(df): {df.shape}")

    return

def load_data(input_file="news_data.csv", input_dir=None, dropna=True, subset=False, n=None, how='any', verbose=False): 
    # np.random.seed(53)

    if input_dir is None: input_dir = os.path.join(os.getcwd(), 'data')
    assert os.path.exists(input_dir)
    
    input_path = os.path.join(input_dir, input_file) 
    df = pd.read_csv(input_path)
    N0 = N = df.shape[0]
    
    # drop rows with null values 
    if dropna:
        df = df.dropna(how=how)
        r_missing = fraction_rows_missing(df, verbose=True)
        assert r_missing == 0, "Found missing values. R(missing)={}".format(r_missing)
        N = df.shape[0]
        if verbose: 
            print(f"> size(df): {N0} -> {N}; dropped {N0-N} rows")
    
    if subset and (n is not None and n > 0): 
        if N > n: 
            df = df.sample(n=n)
        # else no-op
        
    if verbose: 
        print(f"(load_data) Dim(df): {df.shape}")
    
    return df


def save_to_npz(X, y=None, output_dir=None, suffix=None, verbose=0):
    # save numpy array as npz file
    from numpy import asarray
    from numpy import savez_compressed

    hasY = False
    if y is not None and len(y) > 0: 
        assert np.array(y).shape[0] == X.shape[0]
        hasY = True
    
    # save to npy file
    if output_dir is None: output_dir = os.getcwd()
    if suffix is not None: 
        Xfn = "X-%s.npz" % suffix
        yfn = "y-%s.npz" % suffix
    else: 
        Xfn = "X.npz"
        yfn = "y.npz"
    
    X_path = os.path.join(output_dir, Xfn)
    y_path = os.path.join(output_dir, yfn)
    savez_compressed(X_path, X)
    if hasY: 
        savez_compressed(y_path, y)

    if verbose: 
        print(f"[I/O] Saving X to:\n{X_path}\n")
        if hasY: print(f"...   Saving y to:\n{y_path}\n")

    return X_path, y_path
### Alias 
save_XY = save_to_npz 

def load_from_npz(input_dir=None, suffix=None): 
    from numpy import load 

    if input_dir is None: input_dir = os.getcwd()
    if suffix is not None: 
        Xfn = "X-%s.npz" % suffix
        yfn = "y-%s.npz" % suffix
    else: 
        Xfn = "X.npz"
        yfn = "y.npz"

    X_path = os.path.join(input_dir, Xfn)
    y_path = os.path.join(input_dir, yfn)
    
    X = load(X_path)['arr_0']
    try: 
        y = load(y_path)['arr_0']
    except: 
        y = []

    return (X, y)
### Alias
load_XY = load_from_npz

def scale(X, scaler=None, **kargs):
    from sklearn import preprocessing
    if scaler is None: 
        return X 

    if isinstance(scaler, str): 
        if scaler.startswith(('stand', 'z')): # standardize, z-score
            std_scale = preprocessing.StandardScaler().fit(X)
            X = std_scale.transform(X)
        elif scaler.startswith('minmax'): 
            minmax_scale = preprocessing.MinMaxScaler().fit(X)
            X = minmax_scale.transform(X)
        elif scaler.startswith("norm"): # normalize
            norm = kargs.get('norm', 'l2')
            copy = kargs.get('copy', False)
            X = preprocessing.Normalizer(norm=norm, copy=copy).fit_transform(X)
    else: 
        try: 
            X = scaler.transform(X)
        except Exception as e: 
            msg = "(scale) Invalid scaler: {}".format(e)
            raise ValueError(msg)
    return X

def toDF(X, y, cols_x, cols_y):
    import pandas as pd
    dfX = DataFrame(X, columns=cols_x)
    dfY = DataFrame(y, columns=cols_y)
    return pd.concat([dfX, dfY], axis=1)

def toXY(df, cols_x=[], cols_y=[], untracked=[], **kargs): 
    """
    Convert a dataframe in to the (X, y)-format, where 
       X is an n x m numpy array with n instances and m variables
       y is an n x 1 numpy array, representing class labels

    Inupt
    -----
        cols_x: explanatory variables
        cols_y: target/dependent variable(s)
        untracked: meta-data

    """
    verbose = kargs.get('verbose', 1)

    # optional operations
    scaler = kargs.pop('scaler', None) # used when scaler is not None (e.g. "standardize")
    
    X = y = None
    if len(untracked) > 0: # untracked variables
        df = df.drop(untracked, axis=1)
    
    if isinstance(cols_y, str): cols_y = [cols_y, ]
    if len(cols_x) > 0:  
        X = df[cols_x].values
        
        cols_y = list(df.drop(cols_x, axis=1).columns)
        y = df[cols_y].values

    else: 
        if len(cols_y) > 0:
            cols_x = list(df.drop(cols_y, axis=1).columns)
            X = df[cols_x].values
            y = df[cols_y].values
        else: 
            if verbose: 
                print("(toXY) Both cols_x and cols_y are empty => Assuming all attributes are variables (n={})".format(df.shape[1]))
            X = df.values
            y = None

    if scaler is not None:
        if verbose: print("(toXY) Scaling X using method:\"{}\"".format(scaler))
        X = scale(X, scaler=scaler, **kargs)
    
    return (X, y, cols_x, cols_y)

def preprocess(text, stemming=False, remove_stopwords=True):
    """Process tweet function.
    Input:
        texts: Collection of texts to be tokenized and cleaned
        stemming: Set to True to apply stemming  
        remove_stopwords: Set to True to remove stop words
    Output:
        tweets_clean: a list of words containing the processed tweet

    Memo
    ----
    1. tf.keras.preprocessing.text.Tokenizer(
            num_words=None,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True, split=' ', char_level=False, oov_token=None,
            document_count=0, **kwargs)

    """
    import nltk, string
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords     # module for stop words that come with NLTK
    from nltk.stem import PorterStemmer   # module for stemming
    # from tensorflow.keras.preprocessing.text import Tokenizer

    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    # tokenizer = Tokenizer() # num_words=num_words, oov_token="<OOV>"
    # tokenizer.fit_on_texts(sentences)
    tokenizer = word_tokenize

    texts_clean = []
    for word in tokenizer(text):
        if remove_stopwords and (word in stopwords_english):
            continue # remove stopwords
        if word in string.punctuation: 
            continue  # remove punctuation
        stem_word = stemmer.stem(word) if stemming else word  # stemming word
        texts_clean.append(stem_word)

    return texts_clean
preprocess_text = preprocess  # alias 

def preprocess_concepts(keywords): 
    # import ast
    return ast.literal_eval(keywords)  # convert the list in string format to an actual list

# def tokenize(sentences): 
#     # tokenize
#     from tensorflow.keras.preprocessing.text import Tokenizer
#     from tensorflow.keras.preprocessing.sequence import pad_sequences
#     from tensorflow.keras.utils import to_categorical
#     from tensorflow.keras import regularizers

#     # clean text 
#     import nltk
#     # download the stopwords for the process_tweet function
#     nltk.download('stopwords')

#     embedding_dim = 300
#     max_length = 100
#     trunc_type='post'
#     padding_type='post'
#     oov_tok = "<OOV>"

#     # sentences = df['text'].values
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(sentences)
#     word_index = tokenizer.word_index
#     vocab_size=len(word_index)
#     sequences = tokenizer.texts_to_sequences(sentences)
#     padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# 
#
#     return

def compute_doc_vec(model, text, method='mean', exclude_oov=True):
    if method == 'mean': 
        return compute_mean_vec(model, text, exclude_oov=exclude_oov)
    assert NotImplementedError, f"(compute_doc_vec) Unknown method: {method}"

def compute_mean_vec(model, text, **kargs):
    """
    Compute the document vector for `text` 

    Assumption
    ----------
    1. OOV tokens are zero vectors 
    """
    v_zero = np.zeros(model.vector_size)
    
    def get_vector(token): 
        try: 
            v = model[token]
        except: 
            v = v_zero 
        return v

    exclude_oov = kargs.get('exclude_oov', True)

    if isinstance(text, str): 
        tokens = preprocess(text, stemming=False, remove_stopwords=True) # optionally remove stopwords and apply stemming 
    else: 
        assert isinstance(text, (list, np.ndarray))
        tokens = text
    
    if len(tokens) == 0: 
        v = v_zero
    else: 
        if exclude_oov: 
            v = v_zero
            n_hit = 0
            for token in tokens: 
                try: 
                    v += model[token]
                    n_hit += 1
                except: 
                    continue
            if n_hit > 1:  # finally take the average
                v = v/n_hit
            # print(f"exclude_oov: n_hit: {n_hit} out of {len(tokens)}")
        else: 
            v = np.mean(np.array([get_vector(token) for token in tokens]), axis=0)

    assert v.shape[0] == model.vector_size

    return v
def compute_doc_concept_vector_v0(model, text, concept):
    """
    Compute the joint vector between text-specific document vector 
    and the word vector associated with a given concept/tag. 
    """
    def get_vector(token): 
        try: 
            v = model[token]
        except: 
            v = np.zeros(model.vector_size)
        return v

    v_doc = compute_doc_vec(model, text) 
    v_concept = get_vector(concept)
    assert v_doc.shape[0] == v_concept.shape[0]
    return np.hstack((v_doc, v_concept))  

def compute_doc_concept_vector(model, text, concept):
    """
    Similar to compute_doc_concept_vector_v0() but maps 
    the (text, concept)-pair to the default zero vector if 
    either the text or the concept is a zero vector, meaning
    that none of the token as a unknown vector represenation
    in the pre-trained model.
    """
    def get_vector(token): 
        try: 
            v = model[token]
        except: 
            v = np.zeros(model.vector_size)
        return v

    v_doc = compute_doc_vec(model, text, exclude_oov=True) 
    v_concept = get_vector(concept)
    assert v_doc.shape[0] == v_concept.shape[0]

    if np.all(v_doc == 0) or np.all(v_concept==0): 
        # return a special zero vector so that we can potentially exclude them in the training data
        return np.zeros(len(v_doc)+len(v_concept))

    return np.hstack((v_doc, v_concept))  

def transform_documents(input_file='news_data.csv', 
           pretrained_w2v="GoogleNews-vectors-negative300.bin", **kargs): 

    # from sklearn.model_selection import train_test_split
    from gensim.models import KeyedVectors
    # import ast 

    # Optional params
    #####################################
    verbose = kargs.pop('verbose', 2)
    tSave = kargs.pop('save', False)
    tTest = kargs.pop('test', False)

    tIncludeConcepts = kargs.get("include_concepts", False) # treat concepts ('keywords') as documents? 

    output_dir = kargs.get('output_dir', os.path.join(os.getcwd(), 'data'))
    data_index = kargs.get('index', 0) 
    #####################################

    df = load_data(input_file=input_file, subset=False, verbose=1 if verbose else 0)

    if tTest: 
        n_test = 3000
        df = df.sample(n=n_test)
        highlight(f"(transform) Test Mode: subsample n={df.shape[0]} instances ...")

    N0 = df.shape[0]
    model = KeyedVectors.load_word2vec_format(pretrained_w2v, binary=True)

    X, y = make_document_training_set(model, texts=df['text'].values, 
            concepts=df['keywords'].values if tIncludeConcepts else [], 
            shuffle=True,
            verbose=verbose)

    if tSave: 
        output_dir = os.path.join(os.getcwd(), 'data')
        identifier = f"doc-{data_index}" if not tTest else f"simple-doc-{data_index}"
        save_to_npz(X, y, output_dir=output_dir, suffix=identifier, verbose=verbose)

    return X, y
    

def transform_and_split(input_file='news_data.csv', 
           pretrained_w2v="GoogleNews-vectors-negative300.bin", train_size=0.8, **kargs): 
    """
    Transform the input data into the vector representation and create its corresponding
    training split and test split. 
    """
    # from sklearn.model_selection import train_test_split
    from gensim.models import KeyedVectors
    # import ast 

    # Optional params
    ###############################################
    n_samples = kargs.pop('n_samples', None)
    verbose = kargs.pop('verbose', 2)
    tSave = kargs.pop('save', False)
    tTest = kargs.pop('test', False)
    tFullPositiveSet = kargs.pop('full_positive_set', True) # set to True to NOT subsample positive concepts

    output_dir = kargs.get('output_dir', os.path.join(os.getcwd(), 'data'))
    data_index = kargs.get('index', 0) 
    ###############################################

    # Load the data and optionally subsample the data
    df = load_data(input_file=input_file, subset=False, verbose=1 if verbose else 0)
    if n_samples is not None: 
        df = df.sample(n=n_samples)
    else: 
        df = df.sample(frac = 1)

    # Parameters for augmented feature set (additional variables that may help with the prediction)
    ###############################################
    hasClusterId = 'cluster_ids' in df.columns  # does df comes with cluster IDs? 

    if tTest: 
        n_test = min(df.shape[0], 3000)
        df = df.sample(n=n_test)
        highlight(f"(transform) Test Mode: subsample n={df.shape[0]} instances ...")

    N0 = df.shape[0]
    model = KeyedVectors.load_word2vec_format(pretrained_w2v, binary=True)

    # create training split and test split 
    mask = np.random.rand(N0) < train_size
    df_train = df[mask]
    df_test = df[~mask]

    # find all concepts/tags 
    concept_set = set() 
    for concept in df['keywords']:
        concept_set.update(preprocess_concepts(concept)) 
    if verbose > 1: 
        print(f"(transform) Found n={len(concept_set)} concepts")

    X_train, y_train, tracked, untracked, text_train, concept_train = \
        make_training_set(model, texts=df_train['text'].values, 
            concepts=df_train['keywords'].values, 
            cluster_ids=df_train['cluster_id'] if hasClusterId else None, 
            concept_set=concept_set, 
            track_meta_data=False, # track the text-concept combination
            shuffle=True,
            full_positive_set=tFullPositiveSet, 
            verbose=verbose)
    X_test, y_test, tracked_test, untracked_test, text_test, concept_test = \
        make_test_set(model, texts=df_test['text'].values, concepts=df_test['keywords'].values, 
                            cluster_ids=df_test['cluster_id'] if hasClusterId else None, 

                            concept_set=concept_set,
                            tracked=tracked, untracked=untracked, 

                            test_unseen_concepts=True, # create test instances only for the concepts/tags unobserved in the training set
                            track_meta_data=False,   # track the text-concept combination

                            full_positive_set=tFullPositiveSet, 
                            shuffle=True,
                            verbose=verbose)

    if tSave: 
        output_dir = os.path.join(os.getcwd(), 'data')
        identifier = f"train-{data_index}" if not tTest else f"simple-train-{data_index}"
        save_to_npz(X_train, y_train, output_dir=output_dir, suffix=identifier, verbose=verbose)
        
        identifier = f"test-{data_index}" if not tTest else f"simple-test-{data_index}"
        save_to_npz(X_test, y_test, output_dir=output_dir, suffix=identifier, verbose=verbose)

    return X_train, X_test, y_train, y_test  # output made consistent to sklearn.model_selection.train_test_split(...)

# alias
transform = transform_and_split 
transform_documents_and_concepts = transform_and_split
#####################################

def load_document_concept_vectors(input_dir=None, data_index=0, test=False):
    if input_dir is None: 
        input_dir = os.path.join(os.getcwd(), 'data')

    identifier = f"train-{data_index}" if not test else f"simple-train-{data_index}"
    X_train, y_train = load_XY(input_dir=input_dir, suffix=identifier)

    identifier = f"test-{data_index}" if not test else f"simple-test-{data_index}"
    X_test, y_test = load_XY(input_dir=input_dir, suffix=identifier)

    return X_train, X_test, y_train, y_test

def load_document_vectors(input_dir=None, data_index=0, test=False): 
    if input_dir is None: 
        input_dir = os.path.join(os.getcwd(), 'data')

    identifier = f"doc-{data_index}" if not test else f"simple-doc-{data_index}"
    X, y = load_XY(input_dir=input_dir, suffix=identifier)
    
    return X, y

def make_document_training_set(model, texts, **kargs): 

    # Optional params 
    #########################################
    concepts = kargs.pop('concepts', [])
    tConcepts = False
    if len(concepts) > 0: 
        assert len(concepts) == len(texts)
        tConcepts = True

    verbose = kargs.pop('verbose', 1)
    tTrackMeta = kargs.pop('track_meta_data', False)
    tShuffle = kargs.pop('shuffle', True)
    tMatchSampleSize = kargs.pop('match_sample_size', True) # set to True to NOT skip any text whose document vector can not be determined

    n_features = kargs.get('n_features', model.vector_size)
    #########################################

    X, y = [], []
    n_skipped = 0
    N = len(texts)
    for i, text in enumerate(texts):
        tokens = preprocess(text, **kargs) # optionally remove stopwords and apply stemming
        
        # Optional (useful if labeling information e.g. topics, tags is available)
        if tConcepts: 
            concept_set = preprocess_concepts(concepts[i])  # concept set; ith text is tagged with these concepts
        
        # Skip the instances of size zero either in text field or concept field (aka keywords or tags)
        if len(tokens) == 0:
            if not tMatchSampleSize: 
                n_skipped += 1 
                continue
            # note that when len(tokens) == 0, the document vector, by default, is the zero vector

        X.append( compute_doc_vec(model, tokens, method='mean') )
        
        if tConcepts: 
            # treat the list of concepts/tags as a document as well
            y.append( compute_doc_vec(model, concept_set, method='mean') )

        if verbose > 1: 
            if i > 0 and i % 1000 == 0: 
                print(f"... Doc Train Set: Completed {i} instances of input text ({i/(N+0.0)*100:.2f}%)")

    X = np.array(X)
    y = np.array(y)

    if tShuffle: 
        if verbose: highlight("(document_vectors) Shuffling the data ...")
        p = np.random.permutation(X.shape[0])
        X = X[p]

        if len(y) > 0: 
            assert len(y) == X.shape[0]
            y = y[p]

    assert X.shape[1] == model.vector_size
    if tMatchSampleSize: assert n_skipped == 0
    if verbose: 
        print(f"(document_vectors) Doc Train Set: {X.shape[0]}; n_features: {X.shape[1]}")
        print(f"...... number of skipped instances n={n_skipped} out of N={N} instances: ratio={n_skipped/(N+0.0)*100:.2f}%")
    return X, y
    

def make_test_set(model, texts, concepts, **kargs):
    from sampling_utils import negative_sample
    from sklearn.utils import shuffle

    # Optional params  # concept
    #########################################
    concept_set = kargs.pop('concept_set', None)
    cluster_ids = kargs.pop('cluster_ids', None)  

    # subsample concepts to reduce training set size
    n_pos_per_text = kargs.pop('n_pos_per_text', 1)
    n_neg_per_text = kargs.pop('n_neg_per_text', 1)

    verbose = kargs.pop('verbose', 1)

    concept_tracked = kargs.pop('tracked', set())  # concepts/tags observed in the training set
    concept_untracked = kargs.pop('untracked', set())  # the remaining concpets not used in making the training data ... 
    # ... and therefore can be used for testing
    assert len(concept_tracked.intersection(concept_untracked)) == 0

    # Test only unobserved concepts in the training set? 
    tTestUnseenConcepts = kargs.pop('test_unseen_concepts', True if len(concept_tracked) else False)
    tTrackMeta = kargs.pop('track_meta_data', False)
    tShuffle = kargs.pop('shuffle', True)
    random_state = kargs.pop('random_state', 53)

    tFullPositiveSet = kargs.pop('full_positive_set', True) # if set to True, won't subsample positive concepts
    if tFullPositiveSet: 
        if verbose: print("(make_test_set) Keeping all positive concepts/tags ...")
        n_pos_per_text = n_neg_per_text = np.inf
    #########################################

    # Find all concepts
    if concept_set is None:  
        # concept set contains all possible concepts in the (test) data but it's not given as priori, then use ... 
        # ... all concepts found in the test split as the total set
        concept_set = set() 
        for concept in concepts:
            concept_set.update(preprocess_concepts(concept)) 
        if verbose: 
            print(f"(make_training_set) Found {len(concept_set)} concepts/tags")   

    Xpos, Xneg = [], []
    X_text, X_concept = [], []
    n_skipped = 0
    N = len(texts)
    test_concept_tracked = set()
    for i, text in enumerate(texts):
        tokens = preprocess(text, **kargs) # optionally remove stopwords and apply stemming
        positive_set = preprocess_concepts(concepts[i])  # positive concept set; ith text is tagged with these concepts

        if tTestUnseenConcepts: # ensure that the test split is associated only with unseen concepts
            positive_set = set(positive_set) - set(concept_tracked)
        
        # skip the instances of size zero either in text field or concept field (aka keywords or tags)
        if len(tokens) == 0 or len(positive_set) == 0:
            n_skipped += 1 
            continue

        negative_set = negative_sample(positive_set, concept_set) # find subset of concepts that are not in the target set
        # ... by default size(negative_set) is equal to size(positive_set)

         # subsample positive concepts and negative concepts to reduce final sample size
        if n_pos_per_text < len(positive_set): 
            positive_set = np.random.choice(list(positive_set), n_pos_per_text, replace=False)
        if n_neg_per_text < len(negative_set): 
            negative_set = np.random.choice(list(negative_set), n_neg_per_text, replace=False)

        # make positive example(s)
        for concept in positive_set: 
            dcv = compute_doc_concept_vector(model, tokens, concept)
            if not np.all(dcv == 0):  # exclude zero vector, in which case, either text or concept is not sufficiently represented
                Xpos.append( dcv )
                test_concept_tracked.add(concept)

                # keep track of the text,concept combination for further analysis
                if tTrackMeta: 
                    X_text.append( text )
                    X_concept.append( concept )

        # make negative example(s)
        for concept in negative_set:  
            assert not concept in positive_set
            dcv = compute_doc_concept_vector(model, tokens, concept)
            if not np.all(dcv == 0): 
                Xneg.append( dcv )

                # keep track of the text,concept combination for further analysis
                if tTrackMeta: 
                    X_text.append( text )
                    X_concept.append( concept )

        if verbose > 1: 
            if i > 0 and i % 1000 == 0: 
                print(f"... Test split: Completed {i} instances of input text ({i/(N+0.0)*100:.2f}%)")

    test_concept_untracked = concept_set - test_concept_tracked
    n_pos = len(Xpos)
    n_neg = len(Xneg)
    X = np.vstack( (np.array(Xpos), np.array(Xneg)) )
    y = np.hstack( (np.repeat(1, n_pos), np.repeat(0, n_neg)) )

    if tShuffle: 
        if verbose: highlight("(transform) Test split: Shuffling the data ...")
        p = np.random.permutation(X.shape[0])
        # X, y = shuffle(X, y, random_state=random_state)
        X, y = X[p], y[p]

        if len(X_text) > 0: 
            X_text=np.array(X_text)
            assert X_text.shape[0] == X.shape[0]
            X_text = X_text[p]
        if len(X_concept) > 0: 
            X_concept=np.array(X_concept)
            assert X_concept.shape[0] == X.shape[0]
            X_concept = X_concept[p]
    if tTrackMeta: 
        assert len(X_text) == X.shape[0]

    assert X.shape[1] == model.vector_size * 2
    if verbose: 
        print(f"(make_test_set) Test set size: {X.shape[0]}; n_features: {X.shape[1]}")
        print(f"... number of concept tracked within the test split: {len(test_concept_tracked)}")
        print(f"... number of untracked concepts within the test split: {len(test_concept_untracked)}")
        print(f"...... number of skipped instances n={n_skipped} out of N={N} instances: ratio={n_skipped/(N+0.0)*100:.2f}%")
    return X, y, test_concept_tracked, test_concept_untracked, X_text, X_concept

def make_training_set(model, texts, concepts, **kargs):
    from sampling_utils import negative_sample

    assert len(texts) == len(concepts)
    # assert isinstance(concepts[0], (list, np.ndarray))

    # Optional params  # concept
    ############################################################
    concept_set = kargs.pop('concept_set', None)
    cluster_ids = kargs.pop('cluster_ids', None)  # Add cluster membership as a feature (todo)

    # subsample concepts to reduce training set size
    n_pos_per_text = kargs.pop('n_pos_per_text', 1)  
    n_neg_per_text = kargs.pop('n_neg_per_text', 1)

    verbose = kargs.pop('verbose', 1)
    tTrackMeta = kargs.pop('track_meta_data', False)
    tShuffle = kargs.pop('shuffle', True)

    tFullPositiveSet = kargs.pop('full_positive_set', True) # if set to True, won't subsample positive concepts
    if tFullPositiveSet: 
        if verbose: print("(make_training_set) Keeping all positive concepts/tags ...")
        n_pos_per_text = n_neg_per_text = np.inf
    ############################################################

    # "Meta" feature paramters (Todo)  
    ############################################################
    
    # for instance, knowing the cluster membership (of the input text) could help
    cluster_ids = kargs.get('cluster_ids', None)
    if cluster_ids is not None: 
        assert isinstance(cluster_ids, (list, np.ndarray))
        assert len(cluster_ids) == len(texts) 
    ############################################################

    # Find all concepts
    if concept_set is None: 
        concept_set = set()
        for concept in concepts:
            concept_set.update(preprocess_concepts(concept)) 
        if verbose: 
            print(f"(make_training_set) Found {len(concept_set)} concepts/tags")        

    Xpos, Xneg = [], []
    X_text, X_concept = [], []  
    n_skipped = n_hit = n_missed = 0
    N = len(texts)
    concept_tracked = set()  # the subset of concepts that comprise the training set
    concept_untracked = set()   # the remaining concpets not used in making training data (and therefore can be used for testing)
    for i, text in enumerate(texts):
        tokens = preprocess(text, **kargs) # optionally remove stopwords and apply stemming
        positive_set = preprocess_concepts(concepts[i])  # positive concept set; ith text is tagged with these concepts
        
        # skip the instances of size zero either in text field or concept field (aka keywords or tags)
        if len(tokens) == 0 or len(positive_set) == 0:
            n_skipped += 1 
            continue

        negative_set = negative_sample(positive_set, concept_set) # find subset of concepts that are not in the target set
        # ... by default, the size of negative concept set is the same as that of the positive

        # subsample positive concepts and negative concepts to reduce final sample size
        if n_pos_per_text < len(positive_set): 
            positive_set = np.random.choice(list(positive_set), n_pos_per_text, replace=False)
        if n_neg_per_text < len(negative_set): 
            negative_set = np.random.choice(list(negative_set), n_neg_per_text, replace=False)

        # make positive example(s)
        for concept in positive_set: 
            dcv = compute_doc_concept_vector(model, tokens, concept)
            if not np.all(dcv == 0): 
                Xpos.append( dcv )
                concept_tracked.add(concept)
                n_hit += 1

                # keep track of the text,concept combination for further analysis
                if tTrackMeta: 
                    X_text.append( text )
                    X_concept.append( concept )
            else: 
                n_missed += 1

        # make negative example(s)
        for concept in negative_set: 
            assert not concept in positive_set

            dcv = compute_doc_concept_vector(model, tokens, concept)
            if not np.all(dcv == 0): 
                Xneg.append( dcv )
                n_hit += 1

                if tTrackMeta: 
                    X_text.append( text )
                    X_concept.append( concept )
            else: 
                n_missed += 1

        if verbose > 1: 
            if i > 0 and i % 1000 == 0: 
                print(f"... Train split: Completed {i} instances of input text ({i/(N+0.0)*100:.2f}%)")
                print(f"... i={i} | positive set: {','.join(list(positive_set))}")
                print(f"...       | negative set: {','.join(list(negative_set))}")

    concept_untracked = concept_set - concept_tracked
    n_pos = len(Xpos)
    n_neg = len(Xneg)
    X = np.vstack( (np.array(Xpos), np.array(Xneg)) )
    y = np.hstack( (np.repeat(1, n_pos), np.repeat(0, n_neg)) )

    if tShuffle: 
        if verbose: highlight("(transform) Train split: Shuffling the data ...")
        p = np.random.permutation(X.shape[0])
        # X, y = shuffle(X, y, random_state=random_state)
        X, y = X[p], y[p]

        if len(X_text) > 0: 
            X_text=np.array(X_text)
            assert X_text.shape[0] == X.shape[0]
            X_text = X_text[p]
        if len(X_concept) > 0: 
            X_concept=np.array(X_concept)
            assert X_concept.shape[0] == X.shape[0]
            X_concept = X_concept[p]

    if tTrackMeta: 
        assert len(X_text) == X.shape[0]
    assert X.shape[1] == model.vector_size * 2
    if verbose: 
        print(f"(make_training_set) Training set size: {X.shape[0]}; n_features: {X.shape[1]}")
        print(f"... number of concept tracked: {len(concept_tracked)}")
        print(f"... number of remaining untracked concepts: {len(concept_untracked)}")
        print(f"...... number of skipped instances n={n_skipped} out of N={N} instances: ratio={n_skipped/(N+0.0)*100:.2f}%")
    return X, y, concept_tracked, concept_untracked, X_text, X_concept


def verify_pretrained_w2v(input_file="news_data.csv", verify_texts=True, verify_concepts=True, save_=True, verbose=1): 
    """
    Input
    -----
       - input_file 
       - verify_texts: Set to True to examine `text` column
       - verify_concepts: Set to True to examine `keywords` column
       - save_: Set to True to save outputs (e.g. output of the concept occurrences, where each keyword is a concept
                associated with the text i.e. news headline)

    Memo
    ----
    1. EDA question: Shall we apply stemming or lemmatization to use GoogleNews pretrained word embeddings? 
       ~> No, because the 'embedding ratio' would be lower if we applied stemming
         'embedding ratio' is the ratio between number of tokens with known embedding and number of tokens 
         of a given text
    
    2. Useful facts for 'text'
        When no stemming applied: 
            Found N=153573 unique tokens. Average hit ratio: 0.9202011604594252
            Among N=153573 tokens, n=104066 of them has known embedding.

        When Porter stemmer applied: 
            Found N=90291 unique tokens. Average hit ratio: 0.7643262692434042
            Among N=90291 tokens, n=20955 of them has known embedding.
    3. Useful facts for 'label'
        
        Found N=82272 unique labels. Average hit ratio: 0.8983321465595778
        Among N=82272 labels, n=45034 of them has known embedding.
        Max: 6, Min: 1, Median: 6.0, n(Empty): 0

    """
    # from gensim.models.word2vec import Word2Vec
    from gensim.models import KeyedVectors
    # import ast 
    from collections import Counter
    
    df = load_data(input_file=input_file, subset=False, verbose=True)
    N0 = df.shape[0]
    pretrained_w2v_binary_file = "GoogleNews-vectors-negative300.bin"
    model = KeyedVectors.load_word2vec_format(pretrained_w2v_binary_file, binary=True)

    if verify_texts: 
        # Shall we apply stemming or lemmatization to use GoogleNews pretrained word embeddings? 
            
        for stemming in [False, True]: 
            highlight(f"Stemming applied? {'Yes' if stemming else 'No'} ...")
            n_w2v_found = 0
            n_embedding_ratios = []  # find, for each input text, the ratio of word embeddings found from the pretrained model
            n_tokens = []
            all_tokens = set()

            test_ids = np.random.choice(np.arange(N0), 20)
            n_empty = 0
            for i, text in enumerate(df['text'].values): 
                tokens = preprocess(text, stemming=stemming) # remove stopwords and optionally applying stemming
                
                if verbose > 1 and (i in test_ids): print(f"... [stemming: {'Yes' if stemming else 'No'}] {text[:50]} ->\n{tokens[:50]}\n")
                
                N = len(tokens); n_tokens.append(N)
                if N == 0: n_empty += 1
                nw = sum(1 for token in tokens if token in model)
                n_embedding_ratios.append(nw/(N+0.0) if N > 0 else 0)
                all_tokens.update(tokens)

            Na = len(all_tokens)
            n = sum(1 for token in all_tokens if token in model)

            if verbose: 
                print(f"... Found N={len(all_tokens)} unique tokens. Average 'hit' ratio per text: {np.mean(n_embedding_ratios)}") 
                print(f"...... Among N={Na} tokens, n={n} of them has known embedding.") 
                print(f"...... Max: {max(n_tokens)}, Min: {min(n_tokens)}, Median: {np.median(n_tokens)}, n(Empty): {n_empty}")
                
            assert len(n_embedding_ratios) == df.shape[0]
    
    if verify_concepts: 
        n_empty = 0
        n_embedding_ratios = []
        n_concepts = []  # number of concepts/labels per text
        # all_concepts = set()
        enrichment = Counter() # keep track # documents/texts that each concept appears

        test_ids = np.random.choice(np.arange(N0), 20)
        for i, keywords in enumerate(df['keywords']): 
            concepts = preprocess_concepts(keywords)  # convert the list in string format to an actual list
            enrichment.update(concepts)
            if verbose > 1 and (i in test_ids): print(f"... concepts (n={len(concepts)}): {concepts}")

            N = len(concepts); n_concepts.append(N)
            if N == 0: n_empty += 1
            nw = sum(1 for concept in concepts if concept in model)   # labels
            n_embedding_ratios.append(nw/(N+0.0) if N > 0 else 0)
            # all_concepts.update(concepts)

        all_concepts = list(enrichment.keys())
        Na = len(all_concepts)
        n = sum(1 for label in all_concepts if label in model)

        if verbose: 
            print(f"... Found N={Na} unique concepts. Average 'hit' ratio per concept list: {np.mean(n_embedding_ratios)}") 
            print(f"... Among N={Na} concepts, n={n} of them has known embedding.") 
            print(f"...... Max: {max(n_concepts)}, Min: {min(n_concepts)}, Median: {np.median(n_concepts)}, n(Empty): {n_empty}")
        
        n_candidates = 10  
        highlight(f"Most enriched concepts:\n{enrichment.most_common(n_candidates)}\n")
        print(f"...      least enriched concepts:\n{enrichment.most_common()[-n_candidates:]}\n") 

        if save_: 
            header = ['concept', 'freq']
            df = DataFrame(list(enrichment.items()),columns=header)
            output_dir = os.path.join(os.getcwd(), 'data')
            fpath = os.path.join(output_dir, "news_data-enrichment.csv")
            if verbose: print(f"[verify] Saving concept statistics to:\n{fpath}\n")
            df.to_csv(fpath, sep=',', index=False, header=True)

    return

# Demo 
#############################################
def demo_preprocess(): 
    import nltk

    # download the stopwords for the preprocess function
    nltk.download('stopwords')
    nltk.download('punkt')   # used by word tokenizer

    sentences = ["You don't really know much about this whole Paleo thing, but how hard could it be, right?! Right. \
                     Just remember that old saying: Curiosity killed the cat.", 

                 'You love my dog!',
                 'Do you think my dog is amazing?'
                  ]

    for i, sentence in enumerate(sentences): 
        tokens = preprocess(sentence)
        print(f"[{i}] {sentence} -> \n{tokens}\n")

    return

def demo_load_data(input_file='news_data.csv'):
    """

    Memo
    ----
    1. n(rows) of news_data.csv: ~381K
    """
    n_samples = 1000

    df = load_data(input_file, subset=True, n=n_samples, verbose=True)
    assert len(df) <= n_samples

    # load the entire data set
    df = load_data(input_file, subset=False, verbose=True)

    return df

def demo_d2v(): 
    from gensim.models import KeyedVectors

    df = load_data(subset=False, verbose=True)
    N0 = df.shape[0]
    pretrained_w2v_binary_file = "GoogleNews-vectors-negative300.bin"
    model = KeyedVectors.load_word2vec_format(pretrained_w2v_binary_file, binary=True)

    n = 10
    for i, (rid, row) in enumerate(df.sample(n=n).iterrows()):
        if i == 0: 
            highlight(f"[{i}] dtype(text) {type(row['text'])}, dtype(keywords): {type(row['keywords'])}", border=2)
        print(f"+ [{i}] {row['text']}")
        labels = preprocess_concepts(row['keywords'])
        print(f"+  ...  {labels}, nlabels: {len(labels)}")
        dv = compute_mean_vec(model, text=row['text'])
        print(f"+ ... dv: {dv[:20]}")
    return

def run_data_pipeline(test_mode=False): 
     
    # Different ways of loading the data sets
    # demo_load_data()

    # Preprocess texts 
    # demo_preprocess()

    # Examine pre-trained word embeddings
    # verify_pretrained_w2v(verify_texts=True, verify_concepts=True)

    # Examine document vectorization
    # demo_d2v()

    highlight("[test] 1. Create document vectors for clustering ...")
    transform_documents(input_file='news_data.csv', 
                   pretrained_w2v="GoogleNews-vectors-negative300.bin", 
                   save=True, test=test_mode)
    X, _ = load_document_vectors(data_index=0, test=test_mode)  
    print(f"> dim(X): {X.shape}")  

    highlight("[test] 2. Create document-concept vectors for (concept) classification ...")
    transform_documents_and_concepts(input_file='news_data.csv', 
                   pretrained_w2v="GoogleNews-vectors-negative300.bin",   
                   train_size=0.6, save=True,

                   n_samples=None, # 200000  
                   full_positive_set=True, 
                   test=test_mode)

    X_train, X_test, y_train, y_test = load_document_concept_vectors(data_index=0, test=test_mode)
    print(f"> dim(X_trian): {X_train.shape}, dim(X_test): {X_test.shape}")

    return

if __name__ == "__main__": 
    run_data_pipeline(test_mode=False)

