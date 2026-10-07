"""Turn hourly series into (past window -> next H hours) samples.

Each sample for a forecast origin t (the first hour to predict) contains:

  x_past   (L, 1 + F): past target values for hours t-L .. t-1 (scaled, and
                       denoised within this window only if DWT is used),
                       plus the calendar features of those hours
  x_future (H, F):     calendar features of the hours being predicted
                       (known in advance, so valid at any horizon)
  y        (H,):       the RAW scaled target for hours t .. t+H-1

The target is never denoised: the model is always scored against what was
actually measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from efs.preprocessing.wavelet import WaveletConfig, denoise_windows


@dataclass
class WindowSet:
    """Samples ready to be wrapped as tensors (all float32)."""

    x_past: np.ndarray       # (n, L, 1 + F)
    x_future: np.ndarray     # (n, H, F)
    y: np.ndarray            # (n, H)
    origins: pd.DatetimeIndex  # first predicted hour of each sample

    def __len__(self) -> int:
        return len(self.y)


def build_input(
    values: np.ndarray,
    calendar: np.ndarray,
    origin_position: int,
    window_hours: int,
    horizon_hours: int,
    wavelet: WaveletConfig | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Model input for ONE forecast origin, using only ``values[:origin_position]``.

    Used for training samples and, later, for recursive forecasting, where
    ``values`` holds actual history followed by the model's own predictions.
    """
    start = origin_position - window_hours
    if start < 0:
        raise ValueError("Not enough history before the origin for the input window")
    if origin_position + horizon_hours > len(calendar):
        raise ValueError("Horizon runs past the end of the calendar features")

    past_values = np.asarray(values[start:origin_position], dtype=float)
    if wavelet is not None:
        past_values = denoise_windows(past_values, wavelet)

    x_past = np.concatenate([past_values[:, np.newaxis], calendar[start:origin_position]], axis=1)
    x_future = calendar[origin_position:origin_position + horizon_hours]
    return x_past.astype(np.float32), x_future.astype(np.float32)


def build_windows(
    values: np.ndarray,
    calendar: np.ndarray,
    index: pd.DatetimeIndex,
    first_origin: pd.Timestamp,
    last_target_hour: pd.Timestamp,
    window_hours: int,
    horizon_hours: int,
    stride_hours: int,
    wavelet: WaveletConfig | None,
) -> WindowSet:
    """All samples whose predicted hours lie in [first_origin, last_target_hour].

    Input windows may reach back before ``first_origin`` (e.g. validation
    samples look back into October). That is legitimate: those hours are
    already in the past at the forecast origin.
    """
    values = np.asarray(values, dtype=float)
    calendar = np.asarray(calendar, dtype=float)

    first_position = max(int(index.get_loc(first_origin)), window_hours)
    last_position = int(index.get_loc(last_target_hour))
    final_origin = last_position - horizon_hours + 1
    if final_origin < first_position:
        raise ValueError("Period too short for one sample with this window and horizon")

    positions = np.arange(first_position, final_origin + 1, stride_hours)

    # Collect raw windows first so the wavelet transform runs on all of them
    # in one vectorized call (each row is still transformed independently).
    raw_windows = np.empty((len(positions), window_hours))
    for row, position in enumerate(positions):
        raw_windows[row] = values[position - window_hours:position]
    if wavelet is not None:
        past_values = denoise_windows(raw_windows, wavelet)
    else:
        past_values = raw_windows

    n_features = calendar.shape[1]
    x_past = np.empty((len(positions), window_hours, 1 + n_features), dtype=np.float32)
    x_future = np.empty((len(positions), horizon_hours, n_features), dtype=np.float32)
    y = np.empty((len(positions), horizon_hours), dtype=np.float32)
    for row, position in enumerate(positions):
        x_past[row, :, 0] = past_values[row]
        x_past[row, :, 1:] = calendar[position - window_hours:position]
        x_future[row] = calendar[position:position + horizon_hours]
        y[row] = values[position:position + horizon_hours]

    return WindowSet(x_past=x_past, x_future=x_future, y=y, origins=index[positions])
