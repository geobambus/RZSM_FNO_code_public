"""Fourier neural operator for observation-event RZSM prediction."""

import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv(nn.Module):
    """One-dimensional spectral convolution over recent SSM events."""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(
            scale
            * torch.randn(
                self.in_channels,
                self.out_channels,
                self.modes,
                dtype=torch.cfloat,
            )
        )

    def forward(self, values):
        transformed = torch.fft.rfft(values)
        sequence_length = values.size(-1)
        available_modes = sequence_length // 2 + 1
        active_modes = min(self.modes, available_modes)
        output = torch.zeros(
            values.shape[0],
            self.out_channels,
            available_modes,
            device=values.device,
            dtype=torch.cfloat,
        )
        output[:, :, :active_modes] = torch.einsum(
            "bix,iox->box",
            transformed[:, :, :active_modes],
            self.weights[:, :, :active_modes],
        )
        return torch.fft.irfft(output, n=sequence_length)


class FNO(nn.Module):
    """Predict RZSM from 32 SSM observation events and static properties."""

    def __init__(
        self,
        modes,
        width,
        num_static_properties,
        input_channels=3,
        dropout_static=0.1,
        dropout_fc=0.1,
    ):
        super().__init__()
        self.modes = int(modes)
        self.width = int(width)
        self.padding = 8
        self.data_mask_conv = nn.Conv1d(input_channels, self.width, 1)
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_properties, 32),
            nn.GELU(),
            nn.Dropout(dropout_static),
            nn.Linear(32, self.width),
        )
        self.spectral_layers = nn.ModuleList(
            [SpectralConv(self.width, self.width, self.modes) for _ in range(4)]
        )
        self.residual_convs = nn.ModuleList(
            [nn.Conv1d(self.width, self.width, 1) for _ in range(4)]
        )
        self.norm = nn.LayerNorm(self.width)
        self.dropout = nn.Dropout(dropout_fc)
        self.fc1 = nn.Linear(self.width * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, dynamic, static):
        grid = self.get_grid(dynamic.shape, dynamic.device)
        dynamic = torch.cat((dynamic, grid), dim=-1).permute(0, 2, 1)
        dynamic = self.data_mask_conv(dynamic)

        static_embedding = self.static_mlp(static)
        dynamic = dynamic + static_embedding.unsqueeze(-1)
        dynamic = F.pad(dynamic, (0, self.padding))
        for spectral, residual in zip(
            self.spectral_layers, self.residual_convs
        ):
            dynamic = F.gelu(spectral(dynamic) + residual(dynamic))

        dynamic = dynamic[..., :-self.padding].permute(0, 2, 1)
        dynamic = self.norm(dynamic)
        repeated_static = static_embedding.unsqueeze(1).expand(
            -1, dynamic.shape[1], -1
        )
        combined = torch.cat((dynamic, repeated_static), dim=-1)
        combined = self.dropout(combined)
        combined = F.gelu(self.fc1(combined))
        combined = self.dropout(combined)
        return self.fc2(combined[:, -1, :])

    @staticmethod
    def get_grid(shape, device):
        grid = torch.linspace(0, 1, shape[1], dtype=torch.float32, device=device)
        return grid.reshape(1, shape[1], 1).repeat(shape[0], 1, 1)
