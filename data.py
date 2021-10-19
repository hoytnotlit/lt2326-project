# process the data 
import csv
import nltk

def get_data_list():
    with open("quotes_dataset.csv") as file:
        spamreader = csv.reader(file, delimiter=',')
        result = [line[0].lower() for line in spamreader] # lower case everything
        return result


def tokenize_data():
    # TODO would it make sense to remove some words (+punctuation?)
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