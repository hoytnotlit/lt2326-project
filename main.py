import data
import train
import test

import torch
from os.path import exists
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate motivational quotes.')
    parser.add_argument('device', help='device to use, cpu or cuda:n')
    parser.add_argument('n_quotes', help='how many quotes to generate')
    parser.add_argument('--retrain', default=False, action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    retrain = args.retrain
    n_quotes = args.n_quotes
    # get data for training, vocab lenght for model parameter 
    # and word-int-word maps & all tokenized sentences for text generation
    input_data, vocab_len, word_to_int, int_to_word, tokenized = data.get()
    # train model
    if (exists("model.pt") == False or retrain == True):
        train.train(input_data, vocab_len, device)
    # load trained model
    model = torch.load("model.pt").to(device)
    # generate quotes
    for i in range(0, n_quotes):
        seed, actual = test.generate_random_seed(word_to_int, tokenized)
        seed_len = len(seed)
        text_quote, int_quote = test.generate_quote(model, seed, int_to_word, device)
        print("generated quote:", text_quote)
        # get cosine similary as evaluation metric
        cos_sim = test.get_cosine_similarity(int_quote, actual, seed_len)
        print("cosine similarity", cos_sim)