"""LSTM architecture for observation-event RZSM prediction."""

import torch
from torch import nn
from torch.nn import functional as F


class LSTM(nn.Module):
    """Predict RZSM from 32 SSM observation events and static properties."""

    def __init__(
        self,
        hidden_size,
        num_static_properties,
        num_layers=2,
        input_channels=2,
        dropout_static=0.1,
        dropout_fc=0.1,
    ):
        super().__init__()
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_properties, 32),
            nn.GELU(),
            nn.Dropout(dropout_static),
            nn.Linear(32, 32),
        )
        self.lstm = nn.LSTM(
            input_size=input_channels + 32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_fc if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_fc)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, dynamic, static):
        static_embedding = self.static_mlp(static)
        repeated_static = static_embedding.unsqueeze(1).expand(
            -1, dynamic.shape[1], -1
        )
        combined = torch.cat((dynamic, repeated_static), dim=-1)
        sequence, _ = self.lstm(combined)
        output = self.dropout(sequence)
        output = F.gelu(self.fc1(output))
        output = self.dropout(output)
        return self.fc2(output[:, -1, :])
