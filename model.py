import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_len, emb_dim = 200, hidden_dim = 256, num_layers = 2, drop_prob = 0.2):
        super(LSTMModel, self).__init__()

        self.emb = nn.Embedding(vocab_len, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.linear = nn.Linear(hidden_dim, vocab_len)      
    
    def forward(self, input_data):
        embeddings = self.emb(input_data)
        output, _ = self.lstm(embeddings)
        output = self.dropout(output)
        output = self.linear(output)
        return output