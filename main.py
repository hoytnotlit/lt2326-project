import data
import train
import test

import torch
from os.path import exists

if __name__ == "__main__":
    # TODO get cuda from args
    device = torch.device('cuda:3')

    # get data for training, vocab lenght for model parameter 
    # and word-int-word maps & all tokenized sentences for text generation
    input_data, vocab_len, word_to_int, int_to_word, tokenized = data.get()
    # train model
    if (exists("model.pt") == False):
        train.train(input_data, vocab_len, device)
    # load trained model
    model = torch.load("model.pt").to(device)
    # generate quotes
    seed, actual = test.generate_random_seed(word_to_int, tokenized)
    seed_len = len(seed)
    text_quote, int_quote = test.generate_quote(model, seed, int_to_word, device)
    print(text_quote)
    # get cosine similary as evaluation metric
    cos_sim = test.get_cosine_similarity(int_quote, actual, seed_len)
    print(cos_sim)