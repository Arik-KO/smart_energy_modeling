"""Forecasting models: LSTM and encoder-only Transformer (TNN).

Both share one interface so training, tuning and forecasting code never
needs to know which one it is using:

    forward(x_past: [B, L, 1+F], x_future: [B, H, F]) -> [B, H]

x_past holds the past target values and their calendar features;
x_future holds the calendar features of the hours to predict.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PredictionHead(nn.Module):
    """MLP that maps (sequence summary, future calendar) to H predictions."""

    def __init__(self, summary_size: int, horizon_hours: int, n_features: int,
                 mlp_units: int, mlp_dropout: float):
        super().__init__()
        # The future calendar is flattened: H hours x F features. For H=24 and
        # F=8 that is 192 numbers telling the head which hours/days it predicts.
        input_size = summary_size + horizon_hours * n_features
        self.layers = nn.Sequential(
            nn.Linear(input_size, mlp_units),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_units, horizon_hours),
        )

    def forward(self, summary: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        future_flat = x_future.flatten(start_dim=1)
        return self.layers(torch.cat([summary, future_flat], dim=1))


class LSTMForecaster(nn.Module):
    """Stacked LSTM; the last hidden state summarizes the input window."""

    def __init__(self, n_features: int, horizon_hours: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.1, mlp_units: int = 128,
                 mlp_dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1 + n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch applies this dropout between stacked layers only.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = PredictionHead(hidden_size, horizon_hours, n_features, mlp_units, mlp_dropout)

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(x_past)
        last_step = outputs[:, -1, :]          # hidden state after the final input hour
        return self.head(last_step, x_future)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sine/cosine position signal (Vaswani et al., 2017).

    Self-attention by itself treats the input as an unordered set; adding a
    unique pattern per position tells the model which hour is which.
    """

    def __init__(self, d_model: int, max_length: int = 2048):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies[: d_model // 2])
        # A buffer moves with the model to the GPU but is not trained.
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1), :]


POOLING_MODES = ("mean", "last", "mean_last")


class TransformerForecaster(nn.Module):
    """Encoder-only Transformer.

    pooling="mean" is the paper's global average pooling. On its own it is
    nearly blind to the most recent hours: averaging a linear embedding over
    time equals embedding the window average, and a low-pass wavelet filter
    keeps that average almost unchanged, so the DWT could not affect the model.
    "mean_last" adds the final (most recent) token, which after self-attention
    carries information about the latest hours.
    """

    def __init__(self, n_features: int, horizon_hours: int, d_model: int = 64,
                 num_heads: int = 4, num_layers: int = 2, ff_dim: int = 128,
                 dropout: float = 0.1, mlp_units: int = 128, mlp_dropout: float = 0.1,
                 pooling: str = "mean_last", max_length: int = 2048):
        super().__init__()
        if pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {POOLING_MODES}, got '{pooling}'")
        self.pooling = pooling
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.input_embedding = nn.Linear(1 + n_features, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_length)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,        # "Add & Normalize" after each sub-layer (paper Fig. 7)
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        summary_size = 2 * d_model if pooling == "mean_last" else d_model
        self.head = PredictionHead(summary_size, horizon_hours, n_features, mlp_units, mlp_dropout)

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        tokens = self.positional_encoding(self.input_embedding(x_past))
        encoded = self.encoder(tokens)
        mean_token = encoded.mean(dim=1)       # global average pooling over time
        last_token = encoded[:, -1, :]         # the most recent hour, after attention
        if self.pooling == "mean":
            summary = mean_token
        elif self.pooling == "last":
            summary = last_token
        else:
            summary = torch.cat([mean_token, last_token], dim=1)
        return self.head(summary, x_future)

class LevelNormalized(nn.Module):
    """Predict deviations from the input window's average level.

    The network sees past values with the window mean subtracted and its
    output is shifted back by that mean:  y_hat = level + f(x - level).
    Without this, the calendar features (8 of 9 inputs) let the network reach
    a low loss while largely ignoring the past values; with it, the recent
    level is carried into the forecast by construction, and the network only
    has to learn the shape around it (the idea behind RevIN, Kim et al. 2022).
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        level = x_past[:, :, 0].mean(dim=1, keepdim=True)          # (B, 1)
        centred = torch.cat([x_past[:, :, :1] - level.unsqueeze(-1), x_past[:, :, 1:]], dim=2)
        return self.inner(centred, x_future) + level

def base_architecture(architecture: str) -> str:
    """'dwt_lstm' -> 'lstm'; the DWT is preprocessing, not part of the network."""
    return architecture.removeprefix("dwt_")


def uses_dwt(architecture: str) -> bool:
    return architecture.startswith("dwt_")


def build_model(architecture: str, n_features: int, horizon_hours: int, params: dict,
                level_normalization: bool = False) -> nn.Module:
    """Create a model from its architecture name and hyperparameters."""
    base = base_architecture(architecture)
    if base == "lstm":
        model = LSTMForecaster(n_features, horizon_hours, **params)
    elif base == "tnn":
        model = TransformerForecaster(n_features, horizon_hours, **params)
    else:
        raise ValueError(f"Unknown architecture '{architecture}'")
    if level_normalization:
        model = LevelNormalized(model)
    return model


def count_parameters(model: nn.Module) -> int:
    total = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            total += parameter.numel()
    return total
