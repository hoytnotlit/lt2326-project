import data
import train
import test

import torch
from os.path import exists

device = torch.device('cuda:0')

# get data for training, vocab lenght for model parameter 
# and word dictionaries & all tokenized sentences for text generation
input_data, vocab_len, word_to_int, int_to_word, tokenized = data.get()
# train model
if (exists("model.pt") == False):
    train.train(input_data, vocab_len)
# load trained model
model = torch.load("model.pt").to(device)
# generate quotes
seed, actual = test.generate_random_seed(word_to_int, tokenized)
text_quote, int_quote = test.generate_quote(model, seed, int_to_word)
# get some evaluation metrics
test.get_cosine_similarity(int_quote, actual)