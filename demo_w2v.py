
import pickle
import nltk
from gensim.models import KeyedVectors

"""


Reference
---------
1. NLP with Vector Space Models
    - Week 3: https://www.coursera.org/learn/classification-vector-spaces-in-nlp/programming/vbHXo/assignment-word-embeddings/lab

"""

# embeddings = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary = True)
# f = open('capitals.txt', 'r').read()
# set_words = set(nltk.word_tokenize(f))
# select_words = words = ['king', 'queen', 'oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country', 'continent', 'petroleum', 'joyful']
# for w in select_words:
#     set_words.add(w)


def load_word_embeddings(pretrained_w2v="GoogleNews-vectors-negative300.bin"): 
    return KeyedVectors.load_word2vec_format(pretrained_w2v, binary = True)


def get_word_embeddings(embeddings, set_words):

    word_embeddings = {}
    # for word in embeddings.vocab:  #  The vocab attribute was removed from KeyedVector in Gensim 4.0.0
    for word in list(embeddings.index_to_key):
        if word in set_words:
            word_embeddings[word] = embeddings[word]
    return word_embeddings

def process_capitals(input_file='capitals.txt'):
    f = open('capitals.txt', 'r').read()
    set_words = set(nltk.word_tokenize(f))
    select_words = words = ['king', 'queen', 'oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country', 'continent', 'petroleum', 'joyful']
    for w in select_words:
        set_words.add(w)   
    return set_words 

def demo_get_word_embeddings(): 
    
    embeddings = load_word_embeddings(pretrained_w2v="GoogleNews-vectors-negative300.bin")
    print(f"[demo] Size(embeddings): {len(embeddings)}, type: {type(embeddings)}")

    set_words = process_capitals(input_file='capitals.txt')
    print(f"... target words:\n{set_words}\n")

    W = get_word_embeddings(embeddings, set_words)
    print(f"... found n={len(W)} embeddings from {len(set_words)} words")

    # save the subset word embeddings 
    # pickle.dump( word_embeddings, open( "word_embeddings_subset.p", "wb" ) )

    return


def get_concept(child1, parent1, child2, embeddings): 
    """
    Adapt get_country() to implement: 

    child1 is to parent1 as child2 is to ?
 
    """
    return parent2

### Special Case
def get_country(city1, country1, city2, embeddings):
    """
    Note that this is a function of special case, where `embeddings` is assumed 
    to be obtained from `word_embeddings_subset.p` (a subset of google news 
    word embeddings focusing on cities and countries)

    word_embeddings = pickle.load(open("word_embeddings_subset.p", "rb"))


    Input:
        city1: a string (the capital city of country1)
        country1: a string (the country of capital1)
        city2: a string (the capital city of country2)
        embeddings: a dictionary where the keys are words and values are their embeddings
    Output:
        countries: a dictionary with the most likely country and its similarity score
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # store the city1, country 1, and city 2 in a set called group
    group = set((city1, country1, city2))

    # get embeddings of city 1
    city1_emb = embeddings[city1]

    # get embedding of country 1
    country1_emb = embeddings[country1]

    # get embedding of city 2
    city2_emb = embeddings[city2]

    # get embedding of country 2 (it's a combination of the embeddings of country 1, city 1 and city 2)
    # Remember: King - Man + Woman = Queen
    vec = country1_emb - city1_emb + city2_emb

    # Initialize the similarity to -1 (it will be replaced by a similarities that are closer to +1)
    similarity = -1

    # initialize country to an empty string
    country = ''

    # loop through all words in the embeddings dictionary
    for word in embeddings.keys():

        # first check that the word is not already in the 'group'
        if word not in group:

            # get the word embedding
            word_emb = embeddings[word]

            # calculate cosine similarity between embedding of country 2 and the word in the embeddings dictionary
            cur_similarity = cosine_similarity(vec, word_emb)

            # if the cosine similarity is more similar than the previously best similarity...
            if cur_similarity > similarity:

                # update the similarity to the new, better similarity
                similarity = cur_similarity

                # store the country as a tuple, which contains the word and the similarity
                country = (word, similarity)

    ### END CODE HERE ###

    return country

def test():

    # load a subset of (target) word embeddings
    demo_get_word_embeddings()

    return


if __name__ == "__main__": 
    test() 
