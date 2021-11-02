import model

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# training hyperparameters
batch_size = 128
learning_rate = 0.01
epochs = 3

def get_batches(input_data):
    # use DataLoader to get batches 
    return DataLoader(input_data, shuffle=True, batch_size=batch_size)

def train(input_data, vocab_len, device):
    batches = get_batches(input_data)
    lstm = model.LSTMModel(vocab_len).to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr = learning_rate) 
    loss_fn = torch.nn.CrossEntropyLoss()

    lstm.train()
    for e in range(0, epochs):
        total_loss = 0
        batch_n = 1
        for i, (inputs, targets) in enumerate(batches):
            inputs, targets = inputs.to(device), targets.to(device) # send tensors to device
            outputs = lstm(inputs)

            outputs = F.softmax(outputs, dim=1) # TODO why in tutorials softmax not there?
            loss = loss_fn(outputs.permute(0, 2, 1), targets) # calculate loss
            total_loss += loss.item()
            print(f'{total_loss/(i+1)} batch n.{batch_n}', end='\r') # print average loss for the epoch
            batch_n += 1

            loss.backward() # compute gradients
            optimizer.step() # update parameters
            optimizer.zero_grad() # reset gradients
        print()
    torch.save(lstm, "model.pt")