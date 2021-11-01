# process the data 
import csv
import nltk
import itertools
import numpy as np

def get_data_list():
    with open("quotes_dataset.csv") as file:
        reader = csv.reader(file, delimiter=',')
        result = [line[0].lower() for line in reader] # lower case everything
        return result


def tokenize_data():
    # TODO would it make sense to remove some words (+punctuation?)
    # TODO lemmatize + delemmatize possible??
    result = []
    quotes = get_data_list()[:2]
    for quote in quotes:
        sentences = nltk.sent_tokenize(quote)
        words = []
        for sentence in sentences:
            words += nltk.word_tokenize(sentence)
        result.append(words)
    return result


def get_n_grams(n = 2):
    result = []
    tokenized = tokenize_data()
    for quote in tokenized:
        quote_n_grams = []
        for i in range(0, len(quote) - (n-1)):
            n_gram = tuple([quote[i+j] for j in range(0, n)])
            quote_n_grams.append(n_gram)
        result.append(quote_n_grams)
    return result

def make_pairs():
    data_list = get_n_grams()
    result = []
    for sentence in data_list:
        for i in range(0,len(sentence)):
            if i < len(sentence) - 1:
                result.append((sentence[i], sentence[i+1]))
    return result

# map each distinct word in the dataset to an integer value
def get_word_maps():
    sentences = itertools.chain.from_iterable(tokenize_data())
    whole_text = " ".join(sentences).split()
    words = set(whole_text)
    
    word_to_int = {}
    # word to integer
    for i, word in enumerate(words):
        word_to_int[word] = i
    
    int_to_word = list(words)

    return int_to_word, word_to_int

# turn data to integers
def get_data_as_np_array():
    _, word_to_int = get_word_maps()
    data = make_pairs()
    result = []
    for pair in data:
        pair_as_int = []
        for n_gram in pair:
            item_as_int = []
            for word in n_gram:
                item_as_int.append(word_to_int[word])
            pair_as_int.append(item_as_int)
        result.append(pair_as_int)
    return np.array(result)