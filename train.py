import model

import torch
from torch.utils.data import DataLoader

device = torch.device('cuda:0')

# training hyperparameters
batch_size = 64
learning_rate = 0.01
epochs = 3

# TODO split training and testing data (+evaluation)

def get_batches(input_data):
    # use DataLoader to get batches 
    return DataLoader(input_data, shuffle=True, batch_size=batch_size)

def train(input_data, vocab_len):
    batches = get_batches(input_data)
    lstm = model.LSTMModel(vocab_len).to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr = learning_rate) 
    loss_fn = torch.nn.CrossEntropyLoss()

    lstm.train()
    for e in range(0, epochs):
        total_loss = 0
        for i, (inputs, targets) in enumerate(batches):
            inputs, targets = inputs.to(device), targets.to(device) # send tensors to device
            outputs = lstm(inputs)

            # TODO why do i need to permute this?
            loss = loss_fn(outputs.permute(0, 2, 1), targets) # calculate loss
            total_loss += loss.item()
            print(total_loss/(i+1), end='\r') # print average loss for the epoch

            loss.backward() # compute gradients
            optimizer.step() # update parameters
            optimizer.zero_grad() # reset gradients
        print()
    torch.save(lstm, "model.pt")