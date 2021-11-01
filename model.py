import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_len, emb_dim = 200, hidden_dim = 256):
        super(LSTMModel, self).__init__()

        self.emb = nn.Embedding(vocab_len, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        # TODO could add dropout layer?
        # TODO try bidirectional LSTM?
        self.linear = nn.Linear(hidden_dim, vocab_len)      
    
    def forward(self, input_data):
        embeddings = self.emb(input_data)
        output, _ = self.lstm(embeddings)
        output = self.linear(output)
        return output