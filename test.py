# test trained model
import torch
import random

from numpy import dot
from numpy.linalg import norm

def generate_quote(model, seed, int_to_word, device):
    sentence_len = 20
    generated = [int_to_word[i] for i in seed]
    while len(generated) < sentence_len:
        model_input = torch.tensor(seed).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(model_input)
        # TODO need softmax?
        output = output.squeeze(0)[-1] # we only need the last word
        top_predictions = output.argsort()[-5:] # select top 5 values
        selected_prediction = int(random.choice(top_predictions)) # select one of the top predictions
        word = int_to_word[selected_prediction]
        generated.append(word)
        seed.append(selected_prediction)
    return " ".join(generated), seed

def generate_random_seed(word_to_int, sentences):
    # get seed from first couple of words in a random sentence
    sentence = random.choice(sentences)
    seed_length = random.randrange(2, 10) # TODO this might break if sentence lenght is less than 10
    seed = sentence[0:seed_length]
    seed_as_int = [word_to_int[word] for word in seed]
    actual_as_int = [word_to_int[word] for word in sentence]
    return seed_as_int, actual_as_int

def get_cosine_similarity(generated, actual, seed_len):
    # remove seed - it distorts the similarity
    generated = generated[seed_len:]
    actual = actual[seed_len:]
    
    # add padding if vectors lengths don't match
    # if len(generated) < len(actual):
    #     difference = len(actual) - len(generated)
    #     generated = generated + ([0] * difference)
    # if len(actual) < len(generated):
    #     difference = len(generated) - len(actual)
    #     actual = actual + ([0] * difference)

    # truncate vectors if their lengths do not match
    if len(generated) < len(actual):
        actual = actual[0:len(generated)]
    if len(actual) < len(generated):
        generated = generated[0:len(actual)]
    return dot(generated, actual)/(norm(generated)*norm(actual))