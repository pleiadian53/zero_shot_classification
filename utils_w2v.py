import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from utils import get_vectors


# Word2Vec utilities
############################################################

"""
Note
----

* Target models: CBOW, ...

* The center words will be represented as one-hot vectors, and the vectors that represent context words are also based on one-hot vectors.
  - To create one-hot word vectors, you can start by mapping each unique word to a unique integer (or index). 
"""

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

# get center word and context words given a window half size C
def get_windows(words, C):
    """
    The first argument of this function is a list of words (or tokens). 
    The second argument, C, is the context half-size.
    """
    i = C
    while i < len(words) - C:   
        center_word = words[i]
        context_words = words[(i - C):i] + words[(i+1):(i+C+1)]
        yield context_words, center_word
        i += 1

def get_dict(data):
    """
    Input:
        K: the number of negative samples
        data: the data you want to pull from
        indices: a list of word indices
    Output:
        word_dict: a dictionary with the weighted probabilities of each word
        word2Ind: returns dictionary mapping the word to its index
        Ind2Word: returns dictionary mapping the index to its word
    """
    words = sorted(list(set(data)))
    n = len(words)
    idx = 0
    # return these correctly
    word2Ind = {}
    Ind2word = {}
    for k in words:
        word2Ind[k] = idx
        Ind2word[idx] = k
        idx += 1
    return word2Ind, Ind2word

def word_to_one_hot_vector(word, word2Ind, V):
    """
    Convert a word in integer (word index) to one-hot vector
    """
    one_hot_vector = np.zeros(V)
    one_hot_vector[word2Ind[word]] = 1
    return one_hot_vector
get_center_wv = word_to_one_hot_vector

def context_words_to_vector(context_words, word2Ind, V):
    """
    To create the vectors that represent context words, one way is to calculate the average of the one-hot vectors 
    representing the individual words.
    """
    context_words_vectors = [word_to_one_hot_vector(w, word2Ind, V) for w in context_words]
    context_words_vectors = np.mean(context_words_vectors, axis=0)
    return context_words_vectors
get_context_wv = context_words_to_vector

# Define the generator function 'get_training_example'
def get_training_example(words, C, word2Ind, V):
    for context_words, center_word in get_windows(words, C):
        yield context_words_to_vector(context_words, word2Ind, V), word_to_one_hot_vector(center_word, word2Ind, V)

# Define the 'relu' function that will include the steps previously seen
def relu(z):
    result = z.copy()
    result[result < 0] = 0

    # np.maximum(0, z)
    return result

# Define the 'softmax' function that will include the steps previously seen
def softmax(z):
    '''
    Inputs: 
        z: output scores from the hidden layer (assumed to be a 2-D array)
    Outputs: 
        yhat: prediction (estimate of y)
    '''
    # Calculate yhat (softmax)
    yhat = np.exp(z)/np.sum(np.exp(z), axis=0) # summing row vectors

    return yhat

def demo_w2v_data_pipeline(): 

    # Define new corpus
    corpus = 'I am happy because I am learning'

    # Print new corpus
    print(f'Corpus:  {corpus}')

    # Save tokenized version of corpus into 'words' variable
    words = tokenize(corpus)

    ##########################################################################################

    # Print 'context_words' and 'center_word' for the new corpus with a 'context half-size' of 2
    for x, y in get_windows(['i', 'am', 'happy', 'because', 'i', 'am', 'learning'], 2):
        print(f'{x}\t{y}')

    # Print output of 'context_words_to_vector' function for context words: 'am', 'happy', 'i', 'am'
     context_words_to_vector(['am', 'happy', 'i', 'am'], word2Ind, V)

    # Print vectors associated to center and context words for corpus
    for context_words, center_word in get_windows(words, 2):  # reminder: 2 is the context half-size
        print(f'Context words:  {context_words} -> {context_words_to_vector(context_words, word2Ind, V)}')
        print(f'Center word:  {center_word} -> {word_to_one_hot_vector(center_word, word2Ind, V)}')
        print()

    return

def demo_nltk(): 
    import re

    with open('shakespeare.txt') as f:  # download from: https://drive.google.com/open?id=1rSxn_IwcvW8Lv1xWygEulU3D0Y5OJNjM
        data = f.read() 

    # Compute the frequency distribution of the words in the dataset (vocabulary)
    fdist = nltk.FreqDist(word for word in data)
    print("Size of vocabulary: ",len(fdist) )
    print("Most frequent tokens: ",fdist.most_common(20) )
    # ~> [('.', 9630), ('the', 1521), ('and', 1394), ('i', 1257), ('to', 1159), ('of', 1093), ('my', 857), 
    #     ('that', 781), ('in', 770), ('a', 752), ('you', 748), ('is', 630), ('not', 559), ('for', 467), ('it', 460), ...

    return

# CBOW utilities
############################################################

"""
Reference 
---------
1. https://drive.google.com/open?id=1rFkwiiX8N7hQ_FwUgUvAAZa-5EE93rhK
"""

# compute_cost: cross-entropy cost function
def compute_cost(y, yhat, batch_size):
    # cost function 
    logprobs = np.multiply(np.log(yhat),y)  # element-wise multiplication
    cost = - 1/batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost

def forward_prop(x, W1, W2, b1, b2):
    '''
    Inputs: 
        x:  average one hot vector for the context 
        W1, W2, b1, b2:  matrices and biases to be learned
     Outputs: 
        z:  output score vector
    '''
    
    ### START CODE HERE (Replace instances of 'None' with your own code) ###
    
    # Calculate h
    h = np.dot(W1, x) + b1   # (N, V) (V, 1) => Nx1
    
    # Apply the relu on h (store result in h)
    h = np.maximum(0, h)
    
    # Calculate z
    z = np.dot(W2, h) + b2
    
    ### END CODE HERE ###

    return z, h

def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):
    '''
    Inputs: 
        x:  average one hot vector for the context 
        yhat: prediction (estimate of y)
        y:  target vector
        h:  hidden vector (see eq. 1)
        W1, W2, b1, b2:  matrices and biases  
        batch_size: batch size 
     Outputs: 
        grad_W1, grad_W2, grad_b1, grad_b2:  gradients of matrices and biases   
    '''
    # Compute l1 as W2^T (Yhat - Y)
    # Re-use it whenever you see W2^T (Yhat - Y) used to compute a gradient
    l1 = np.dot(W2.T, yhat-y)    # VxN -> NxV Vxm -> Nxm
    # Apply relu to l1
    l1 = np.maximum(0, l1)  # element-wise max(0, l1)
    # Compute the gradient of W1
    grad_W1 = 1/batch_size * np.dot(l1, x.T)   # W1: NxV ~ Nxm mxV => NxV
    # Compute the gradient of W2
    grad_W2 = 1/batch_size * np.dot(yhat-y, h.T) # W2: VxN  ~ Vxm  (Nxm -> mxN) -> VxN
    # Compute the gradient of b1
    grad_b1 = 1/batch_size * np.sum(l1, axis=1, keepdims=True)         # Nxm -> Nx1
    # Compute the gradient of b2
    grad_b2 = 1/batch_size * np.sum(yhat-y, axis=1, keepdims=True)    # Vxm -> Nx1
    # print(f"... dim(grad_b2): {grad_b2.shape}")
    
    return grad_W1, grad_W2, grad_b1, grad_b2

def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03):
    
    '''
    This is the gradient_descent function
    
      Inputs: 
        data:      text
        word2Ind:  words to Indices
        N:         dimension of hidden vector  
        V:         dimension of vocabulary 
        num_iters: number of iterations  
     Outputs: 
        W1, W2, b1, b2:  updated matrices and biases   

    '''
    W1, W2, b1, b2 = initialize_model(N,V, random_seed=282)
    batch_size = 128
    iters = 0
    C = 2
    for x, y in get_batches(data, word2Ind, V, C, batch_size):
        ### START CODE HERE (Replace instances of 'None' with your own code) ###
        # Get z and h
        z, h = forward_prop(x, W1, W2, b1, b2)
        # Get yhat
        yhat = softmax(z)
        # Get cost
        cost = compute_cost(y, yhat, batch_size)
        if ( (iters+1) % 10 == 0):
            print(f"iters: {iters + 1} cost: {cost:.6f}")
        # Get gradients
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)
        
        # Update weights and biases
        W1 = W1 - alpha * grad_W1 
        W2 = W2 - alpha * grad_W2
        b1 = b1 - alpha * grad_b1
        b2 = b2 - alpha * grad_b2
        
        ### END CODE HERE ###
        
        iters += 1 
        if iters == num_iters: 
            break
        if iters % 100 == 0:
            alpha *= 0.66
            
    return W1, W2, b1, b2

def extract_embeddings(words, W1, W2, word2Ind): 
    embs = (W1.T + W2)/2.0  # VxN
     
    # given a list of words and the embeddings, it returns a matrix with all the embeddings
    idx = [word2Ind[word] for word in words]
    X = embs[idx, :]
    print(X.shape, idx)  # X.shape:  Number of words of dimension N each 
    return X

def visualize_embeddigs(X): 
    from matplotlib import pyplot

    result= compute_pca(X, 2)
    pyplot.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()

    return

############################################################

def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''
    dot = np.dot(A, B)
    norma = np.linalg.norm(A, 2)
    normb = np.linalg.norm(B, 2)
    cos = dot/(norma*normb)

    return cos

def euclidean(A, B):
    """
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        d: numerical number representing the Euclidean distance between A and B.
    """
    # euclidean distance

    d = np.linalg.norm(A-B, 2)

    return d

def compute_pca(X, n_components=2):
    """
    Input:
        X: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output:
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    """

    # mean center the data
    X_demeaned = X - np.mean(X, axis=0)

    # calculate the covariance matrix
    covariance_matrix = np.cov(X_demeaned, rowvar=False)

    # calculate eigenvectors & eigenvalues of the covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix)
    # print(f"eigv: {eigen_vals}")

    # sort eigenvalue in increasing order (get the indices from the sort)
    idx_sorted = np.argsort(eigen_vals) # [::-1]
    
    # reverse the order so that it's from highest to lowest.
    idx_sorted_decreasing = idx_sorted[::-1]

    # sort the eigen values by idx_sorted_decreasing
    eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]
    # print(f"eigv (sorted): {eigen_vals_sorted}")

    # sort eigenvectors using the idx_sorted_decreasing indices
    eigen_vecs_sorted = eigen_vecs[:, idx_sorted_decreasing] # NOTE that eigen_vecs is in column-vector format!

    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    eigen_vecs_subset = eigen_vecs_sorted[:, :n_components]  # select first n column vectors
    # print(f"dim(R): {eigen_vecs_subset.shape}, dim(Xm): {X_demeaned.shape}")

    # transform the data by multiplying the transpose of the eigenvectors 
    # with the transpose of the de-meaned data
    # Then take the transpose of that product.
    X_reduced = np.matmul(eigen_vecs_subset.T, X_demeaned.T).T    # (2, 10) * (10, 3) -> (2, 3) -> (3, 2)


    return X_reduced

# Procedure to plot and arrows that represents vectors with pyplot
def plot_vectors(vectors, colors=['k', 'b', 'r', 'm', 'c'], axes=None, fname='image.svg', ax=None):
    """


    Reference
    ---------
    1. Adpated from utils_nb.py 

       https://drive.google.com/file/d/1dN45me-q6WU6CntCJBIltJmQN4Xu9eZB/view?usp=sharing
    """

    scale = 1
    scale_units = 'x'
    x_dir = []
    y_dir = []
    
    for i, vec in enumerate(vectors):
        x_dir.append(vec[0][0])
        y_dir.append(vec[0][1])
    
    if ax == None:
        fig, ax2 = plt.subplots()
    else:
        ax2 = ax
      
    if axes == None:
        x_axis = 2 + np.max(np.abs(x_dir))
        y_axis = 2 + np.max(np.abs(y_dir))
    else:
        x_axis = axes[0]
        y_axis = axes[1]
        
    ax2.axis([-x_axis, x_axis, -y_axis, y_axis])
        
    for i, vec in enumerate(vectors):
        ax2.arrow(0, 0, vec[0][0], vec[0][1], head_width=0.05 * x_axis, head_length=0.05 * y_axis, fc=colors[i], ec=colors[i])
    
    if ax == None:
        plt.show()
        fig.savefig(fname)


def demo_plot_vector(): 
    import matplotlib.pyplot as plt        # Import matplotlib for charts
    # from utils_nb import plot_vectors      # Function to plot vectors (arrows)
    
    x = np.array([[1, 1]]) # Create a 1 x 2 matrix

    plot_vectors([x], axes=[4, 4], fname='transform_x.svg')

    # Create a 2 x 2 (rotation) matrix
    R = np.array([[2, 0],
                  [0, -2]])
    y = np.dot(x, R) # Apply the dot product between x and R

    plot_vectors([x, y], axes=[4, 4], fname='transformx_and_y.svg') # notice how y is a rotated (and scaled) version of x
    
    return

def test(): 

    return

if __name__ == "__main__":
    test() 