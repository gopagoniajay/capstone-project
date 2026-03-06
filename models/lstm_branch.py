import torch
import torch.nn as nn

class LSTMBranch(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super(LSTMBranch, self).__init__()
        self.hidden_size = hidden_size
        
        # LSTM for temporal dependency learning
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        output, (h_n, c_n) = self.lstm(x)
        # Use the final hidden state or average pooling of outputs
        # return h_n[-1] # Shape: [batch, hidden_size]
        
        return output # Return full sequence [batch, seq_len, hidden_size] layer for Attention
