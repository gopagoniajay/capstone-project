import torch
import torch.nn as nn
from models.cnn_branch import CNNBranch
from models.lstm_branch import LSTMBranch

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, encoder_outputs):
        # encoder_outputs: [batch, seq_len, hidden_size]
        weights = self.attn(encoder_outputs) # [batch, seq_len, 1]
        weights = torch.softmax(weights, dim=1)
        
        # Weighted sum
        context = torch.sum(encoder_outputs * weights, dim=1) # [batch, hidden_size]
        return context

class EnsembleModel(nn.Module):
    def __init__(self, num_classes=3):
        super(EnsembleModel, self).__init__()
        
        self.cnn = CNNBranch()
        self.lstm = LSTMBranch()
        self.attention = SelfAttention(hidden_size=64)
        
        # Combine features: 64 (CNN) + 64 (LSTM+Attn)
        self.classifier = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.5), # Increased dropout for regularization
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        cnn_out = self.cnn(x)
        lstm_out_seq = self.lstm(x) # Now returns [batch, seq, 64]
        
        # Apply Attention
        lstm_context = self.attention(lstm_out_seq) # [batch, 64]
        
        # Concatenate features
        combined = torch.cat((cnn_out, lstm_context), dim=1)
        
        return self.classifier(combined)
