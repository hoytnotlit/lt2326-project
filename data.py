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


def tokenize_data(quotes):
    print("tokenizing data...")
    # TODO would it make sense to remove some words (+punctuation?)
    # TODO lemmatize + delemmatize possible??
    result = []
    for quote in quotes:
        sentences = nltk.sent_tokenize(quote)
        words = []
        for sentence in sentences:
            words += nltk.word_tokenize(sentence)
        result.append(words)
    return result


def get_n_grams(tokenized, n = 5):
    print("generating n-grams...")

    result = []
    for quote in tokenized:
        quote_n_grams = []
        for i in range(0, len(quote) - (n-1)):
            n_gram = tuple([quote[i+j] for j in range(0, n)])
            quote_n_grams.append(n_gram)
        result.append(quote_n_grams)
    return result

def make_pairs(data_list):
    print("generating input-target pairs...")

    result = []
    for sentence in data_list:
        for i in range(0,len(sentence)):
            if i < len(sentence) - 1:
                result.append((sentence[i], sentence[i+1]))
    return result

# map each distinct word in the dataset to an integer value
def get_word_maps(tokenized_data):
    print("generating word maps...")

    sentences = itertools.chain.from_iterable(tokenized_data)
    whole_text = " ".join(sentences).split()
    words = set(whole_text)
    
    word_to_int = {}
    # word to integer
    for i, word in enumerate(words):
        word_to_int[word] = i
    
    int_to_word = list(words)
    return int_to_word, word_to_int, len(int_to_word) # return vocab length as well

# turn data to integers
def get_data(word_to_int, data):
    print("generating data to integer arrays...")

    result = []
    for pair in data:
        pair_as_int = []
        for n_gram in pair:
            item_as_int = []
            for word in n_gram:
                item_as_int.append(word_to_int[word])
            pair_as_int.append(np.array(item_as_int))
        result.append(tuple(pair_as_int))
    return result

def get():
    data_list = get_data_list()
    data_list = data_list[:1000] # truncating data because the whole set is too much for CUDA
    tokenized = tokenize_data(data_list)
    n_grams = get_n_grams(tokenized, n=10)
    pairs = make_pairs(n_grams)
    int_to_word, word_to_int, vocab_len = get_word_maps(tokenized)
    result = get_data(word_to_int, pairs)
    return result, vocab_len, word_to_int, int_to_word, tokenized
