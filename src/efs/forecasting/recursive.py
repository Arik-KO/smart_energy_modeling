"""Producing forecasts without ever reading actual values after the origin.

recursive_forecast: from one origin, predict H hours, write the predictions
into the history, move forward, repeat. Used for month-ahead (one origin) and
week-ahead (one origin per week, history refreshed with actuals each time).

one_step_forecast: every prediction uses the actual preceding hours
(the 1-hour-ahead comparison with [12]); clearly NOT a multi-day forecast.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from efs.forecasting.windowing import build_input
from efs.preprocessing.wavelet import WaveletConfig


@torch.no_grad()
def recursive_forecast(
    model: nn.Module,
    values: np.ndarray,
    calendar: np.ndarray,
    origin_position: int,
    total_hours: int,
    window_hours: int,
    horizon_hours: int,
    wavelet: WaveletConfig | None,
    device: torch.device,
    clip_min: float | None = 0.0,
) -> np.ndarray:
    """Forecast ``total_hours`` hours from ``origin_position`` (scaled units).

    Everything from the origin onwards is overwritten with NaN before the loop,
    so any accidental use of an actual future value fails loudly instead of
    silently improving the forecast.
    """
    model.eval()
    history = np.asarray(values, dtype=float).copy()
    history[origin_position:] = np.nan

    predictions = np.empty(total_hours)
    step = 0
    while step < total_hours:
        position = origin_position + step
        x_past, x_future = build_input(history, calendar, position, window_hours,
                                       horizon_hours, wavelet)
        if np.isnan(x_past).any():
            raise RuntimeError("Input window contains unfilled future values")

        output = model(
            torch.from_numpy(x_past).unsqueeze(0).to(device),
            torch.from_numpy(x_future).unsqueeze(0).to(device),
        )
        block = output.squeeze(0).cpu().numpy().astype(float)
        if clip_min is not None:
            # Energy cannot be negative; scaled 0 is the training minimum.
            block = np.maximum(block, clip_min)

        n_keep = min(horizon_hours, total_hours - step)
        predictions[step:step + n_keep] = block[:n_keep]
        # Feed the predictions back as if they were observations.
        history[position:position + n_keep] = block[:n_keep]
        step += n_keep

    return predictions


def rolling_origin_forecast(
    model: nn.Module,
    values: np.ndarray,
    calendar: np.ndarray,
    origin_positions: list[int],
    end_position: int,
    window_hours: int,
    horizon_hours: int,
    wavelet: WaveletConfig | None,
    device: torch.device,
) -> np.ndarray:
    """Week-ahead style: a fresh recursive forecast from each origin.

    Before each origin the history holds ACTUAL values up to that origin
    (the week that just ended has been observed). Each forecast runs until the
    next origin, or ``end_position`` (exclusive) for the last one.
    """
    pieces = []
    boundaries = list(origin_positions) + [end_position]
    for origin, next_origin in zip(boundaries[:-1], boundaries[1:]):
        pieces.append(recursive_forecast(model, values, calendar, origin, next_origin - origin,
                                         window_hours, horizon_hours, wavelet, device))
    return np.concatenate(pieces)


@torch.no_grad()
def one_step_forecast(
    model: nn.Module,
    values: np.ndarray,
    calendar: np.ndarray,
    start_position: int,
    total_hours: int,
    window_hours: int,
    wavelet: WaveletConfig | None,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """1-hour-ahead forecasts using actual preceding hours (model must have H=1)."""
    model.eval()
    x_past_list = []
    x_future_list = []
    for offset in range(total_hours):
        x_past, x_future = build_input(values, calendar, start_position + offset,
                                       window_hours, 1, wavelet)
        x_past_list.append(x_past)
        x_future_list.append(x_future)
    x_past_all = torch.from_numpy(np.stack(x_past_list))
    x_future_all = torch.from_numpy(np.stack(x_future_list))

    outputs = []
    for start in range(0, total_hours, batch_size):
        batch_past = x_past_all[start:start + batch_size].to(device)
        batch_future = x_future_all[start:start + batch_size].to(device)
        outputs.append(model(batch_past, batch_future).squeeze(1).cpu().numpy())
    return np.maximum(np.concatenate(outputs).astype(float), 0.0)


def seasonal_naive_forecast(values: np.ndarray, origin_position: int, total_hours: int,
                            season_hours: int = 168) -> np.ndarray:
    """Repeat the last observed week (the standard benchmark for hourly load)."""
    last_season = np.asarray(values[origin_position - season_hours:origin_position], dtype=float)
    repeats = int(np.ceil(total_hours / season_hours))
    return np.tile(last_season, repeats)[:total_hours]
