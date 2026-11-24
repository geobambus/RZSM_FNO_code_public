"""The event FNO used by the published analyses.

Checkpoint names and initialization order match the trained 32-event models.
The rainfall channel and internal event coordinate retain their saved schema.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FNO_EVENT_DEFAULT_PADDING = 8


class SpectralConv(nn.Module):

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))


    def forward(self, x):
        batch_size = x.shape[0]
        sequence_length = x.size(-1)
        available_modes = sequence_length // 2 + 1
        active_modes = min(self.modes, available_modes)
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batch_size, self.out_channels, available_modes,
            device=x.device, dtype=torch.cfloat,
        )
        mixed = torch.einsum(
            'bix,iox->box', x_ft[:, :, :active_modes],
            self.weights[:, :, :active_modes],
        )
        out_ft[:, :, :active_modes] = mixed
        return torch.fft.irfft(out_ft, n=sequence_length)


class FNOEvent(nn.Module):
    """Sequence-to-one FNO over irregular observation events.

    ``x_dynamic`` contains SSM and scaled time gaps, plus rainfall when enabled.
    A normalized event index is appended internally, preserving the trained
    checkpoint schema.
    """

    def __init__(
        self, modes, width, num_static_properties=None, input_channels=3,
        dropout_static=0.1, dropout_fc=0.1, *,
        num_static_props=None, in_channels=None,
        padding_events=FNO_EVENT_DEFAULT_PADDING, include_precipitation=False,
    ):
        super().__init__()
        if num_static_properties is None:
            num_static_properties = num_static_props
        elif num_static_props is not None and num_static_props != num_static_properties:
            raise ValueError('Conflicting static-feature counts were provided.')
        if num_static_properties is None:
            raise TypeError('num_static_properties is required')
        if in_channels is not None:
            if input_channels != 3 and in_channels != input_channels:
                raise ValueError('Conflicting dynamic-channel counts were provided.')
            input_channels = in_channels
        self.include_precipitation = bool(include_precipitation)
        expected_channels = 4 if self.include_precipitation else 3
        if int(input_channels) != expected_channels:
            raise ValueError(f'Event FNO requires {expected_channels - 1} dynamic channels plus one internal event coordinate.')
        if self.include_precipitation:
            self.register_buffer('precipitation_channel_version', torch.tensor(1))
        self.modes = int(modes)
        self.width = int(width)
        self.padding = int(padding_events)
        if self.padding < 1:
            raise ValueError('padding_events must be positive')
        if self.padding != FNO_EVENT_DEFAULT_PADDING:
            self.register_buffer('padding_events', torch.tensor(self.padding))
        self.num_blocks = 4
        self.data_mask_conv = nn.Conv1d(in_channels=3, out_channels=self.width, kernel_size=1)
        if self.include_precipitation:
            original_lift = self.data_mask_conv
            with torch.random.fork_rng(devices=[]):
                self.data_mask_conv = nn.Conv1d(4, self.width, 1)
            with torch.no_grad():
                self.data_mask_conv.weight[:, :2].copy_(original_lift.weight[:, :2])
                self.data_mask_conv.weight[:, 2].zero_()
                self.data_mask_conv.weight[:, 3].copy_(original_lift.weight[:, 2])
                self.data_mask_conv.bias.copy_(original_lift.bias)
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_properties, 32), nn.GELU(),
            nn.Dropout(dropout_static), nn.Linear(32, self.width),
        )
        self.spectral_layers = nn.ModuleList(
            [SpectralConv(self.width, self.width, self.modes) for _ in range(self.num_blocks)])
        self.residual_convs = nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(self.num_blocks)])
        self.norm = nn.LayerNorm(self.width)
        self.dropout = nn.Dropout(dropout_fc)
        head_width = self.width * 2
        self.fc1 = nn.Linear(head_width, 64)
        self.fc2 = nn.Linear(64, 1)


    def encode_events(self, x_dynamic, x_static):
        """Encode supplied past observations without changing their support."""
        expected_dynamic = 3 if self.include_precipitation else 2
        if x_dynamic.ndim != 3 or x_dynamic.shape[-1] != expected_dynamic:
            raise ValueError(f'Event FNO input must have shape (batch, events, {expected_dynamic}).')
        grid = self.get_grid(x_dynamic.shape, x_dynamic.device)
        x_dyn = torch.cat((x_dynamic, grid), dim=-1).permute(0, 2, 1)
        x_dyn = self.data_mask_conv(x_dyn)
        x_stat_raw = self.static_mlp(x_static)
        x_dyn = x_dyn + x_stat_raw.unsqueeze(-1)
        x_dyn = F.pad(x_dyn, (0, self.padding))
        for spectral, residual in zip(self.spectral_layers, self.residual_convs):
            x_dyn = F.gelu(spectral(x_dyn) + residual(x_dyn))
        x_dyn = x_dyn[..., :-self.padding].permute(0, 2, 1)
        x_dyn = self.norm(x_dyn)
        return (x_dyn, x_stat_raw)


    def forward(self, x_dynamic, x_static):
        x_dyn, x_stat_raw = self.encode_events(x_dynamic, x_static)
        x_stat = x_stat_raw.unsqueeze(1).expand(-1, x_dyn.shape[1], -1)
        features = torch.cat((x_dyn, x_stat), dim=-1)
        features = self.dropout(features)
        features = F.gelu(self.fc1(features))
        features = self.dropout(features)
        return self.fc2(features[:, -1, :])


    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        key = prefix + 'precipitation_channel_version'
        if self.include_precipitation and key in state_dict:
            value = state_dict[key]
            if value.ndim != 0 or value.item() != 1:
                error_msgs.append(f'{key} must record precipitation channel version 1')
        key = prefix + 'padding_events'
        if self.padding != FNO_EVENT_DEFAULT_PADDING and key in state_dict:
            if int(state_dict[key]) != self.padding:
                error_msgs.append(f'{key} must record padding {self.padding}')
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


    @staticmethod
    def get_grid(shape, device):
        grid = torch.linspace(0, 1, shape[1], dtype=torch.float32, device=device)
        return grid.reshape(1, shape[1], 1).repeat(shape[0], 1, 1)
