"""Sequence-to-one LSTM for the 32-observation event recipe."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEvent(nn.Module):
    """Sequence-to-one LSTM for ``[SSM, delta_t_days_scaled]`` events."""

    def __init__(
        self,
        hidden_size,
        num_static_properties=None,
        num_layers=2,
        input_channels=2,
        dropout_static=0.1,
        dropout_fc=0.1,
        *,
        num_static_props=None,
        in_channels=None,
        include_precipitation=False,
    ):
        super().__init__()
        if num_static_properties is None:
            num_static_properties = num_static_props
        elif (
            num_static_props is not None
            and num_static_props != num_static_properties
        ):
            raise ValueError("Conflicting static-feature counts were provided.")
        if num_static_properties is None:
            raise TypeError("num_static_properties is required")
        if in_channels is not None:
            if input_channels != 2 and in_channels != input_channels:
                raise ValueError("Conflicting dynamic-channel counts were provided.")
            input_channels = in_channels
        expected_channels = 3 if include_precipitation else 2
        if int(input_channels) != expected_channels:
            raise ValueError(f"Event LSTM requires {expected_channels} dynamic channels for this recipe.")
        self.input_channels = expected_channels
        if include_precipitation:
            self.register_buffer("precipitation_channel_version", torch.tensor(1, dtype=torch.int64))

        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_properties, 32),
            nn.GELU(),
            nn.Dropout(dropout_static),
            nn.Linear(32, 32),
        )
        self.LSTM = nn.LSTM(
            input_size=32 + self.input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_fc if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_fc)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)


    def forward(self, x_dynamic, x_static):
        if x_dynamic.ndim != 3 or x_dynamic.shape[-1] != self.input_channels:
            raise ValueError(
                f"Event LSTM input must have shape (batch, events, {self.input_channels})."
            )
        x_stat = self.static_mlp(x_static)
        x_stat = x_stat.unsqueeze(1).expand(-1, x_dynamic.shape[1], -1)
        combined = torch.cat((x_dynamic, x_stat), dim=-1)
        recurrent, _ = self.LSTM(combined)
        features = self.dropout(recurrent[:, -1, :])
        features = F.gelu(self.fc1(features))
        features = self.dropout(features)
        return self.fc2(features)
