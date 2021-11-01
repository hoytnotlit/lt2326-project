# TODO test trained model
import torch
import random
import data

def generate_quote(model, seed, int_to_word):
    sentence_len = 20
    generated = [int_to_word[i] for i in seed]
    while len(generated) < sentence_len:
        with torch.no_grad():
            output = model(seed)
        # TODO need softmax?
        output = output[-1] # we only need the last word
        top_predictions = output.argsort()[-5:][::-1] # select top 5 values
        selected_prediction = random.choice(top_predictions) # select one of the top predictions
        #sampled_token_index = top_n_idx[random.sample([0,1,2],1)[0]]
        word = int_to_word[int(selected_prediction)]
        generated.append(word)
        seed += output
    return " ".join(generated), seed

def generate_random_seed(word_to_int, sentences):
    # get seed from first couple of words in a random sentence
    sentence = random.choice(sentences)
    seed_length = random.randrange(2, 10) # TODO this might break if sentence lenght is less than 10
    seed = sentence[0:seed_length]
    seed_as_int = data.get_data(word_to_int, seed)
    actual_as_int = data.get_data(word_to_int, sentence)
    return seed_as_int, actual_as_int

def get_cosine_similarity():
    pass