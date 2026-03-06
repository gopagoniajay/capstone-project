import torch
import torch.nn as nn

class CNNBranch(nn.Module):
    def __init__(self, input_channels=2, output_size=64):
        super(CNNBranch, self).__init__()
        # 1D CNN for spatial feature extraction from time-series
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), # Flattens to [batch, 64, 1]
            nn.Flatten()             # [batch, 64]
        )
        self.output_size = output_size

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        # Conv1d expects: [batch, features, seq_len]
        x = x.transpose(1, 2) 
        return self.network(x)
