"""Tests for train-only scaling, causal wavelet denoising and windowing.

The key test is the leakage test: changing any value at or after a forecast
origin must not change the model's input for that origin.
"""

import numpy as np
import pandas as pd
import pytest

from efs.forecasting.windowing import build_input, build_windows
from efs.preprocessing.scaling import MinMaxScaler1D
from efs.preprocessing.wavelet import WaveletConfig, check_level, denoise_windows

DB4 = WaveletConfig(wavelet="db4", level=3, mode="symmetric", method="approximation")


def synthetic_series(n_hours: int = 24 * 60, seed: int = 0):
    """A daily cycle plus noise, with simple calendar features."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-01", periods=n_hours, freq="h")
    hours = np.arange(n_hours)
    values = 1.0 + 0.5 * np.sin(2 * np.pi * hours / 24) + 0.1 * rng.standard_normal(n_hours)
    calendar = np.column_stack([np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)])
    return index, values, calendar


def test_scaler_uses_training_data_only():
    train = np.array([1.0, 2.0, 3.0])
    scaler = MinMaxScaler1D.fit(train)
    # A larger test value maps above 1 instead of changing the scaler.
    assert np.allclose(scaler.transform([1.0, 3.0, 5.0]), [0.0, 1.0, 2.0])
    assert np.allclose(scaler.inverse_transform(scaler.transform([2.5])), [2.5])


def test_approximation_removes_noise_but_keeps_shape():
    # Level 2 keeps cycles longer than about 8 h, so the daily cycle survives
    # while hour-to-hour noise is removed. (At level 3 the cutoff moves close to
    # the 24 h cycle and starts distorting it; the DWT ablation measures this.)
    _, values, _ = synthetic_series()
    window = values[:336]
    smooth = denoise_windows(window, WaveletConfig(wavelet="db4", level=2))
    clean = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(336) / 24)
    # Denoised signal is closer to the clean daily cycle than the noisy input.
    assert np.abs(smooth - clean).mean() < np.abs(window - clean).mean()
    assert smooth.shape == window.shape


def test_soft_threshold_runs_per_window():
    _, values, _ = synthetic_series()
    windows = np.stack([values[:336], values[336:672]])
    cfg = WaveletConfig(wavelet="sym4", level=2, method="soft_threshold")
    out = denoise_windows(windows, cfg)
    assert out.shape == windows.shape
    # Each row is transformed on its own: row 0 does not depend on row 1.
    alone = denoise_windows(windows[0], cfg)
    assert np.allclose(out[0], alone)


def test_level_too_deep_is_rejected():
    with pytest.raises(ValueError):
        check_level(32, WaveletConfig(wavelet="db4", level=6))


@pytest.mark.parametrize("wavelet", [None, DB4])
def test_no_future_leakage(wavelet):
    index, values, calendar = synthetic_series()
    origin = 24 * 30
    x_past, _ = build_input(values, calendar, origin, 336, 24, wavelet)

    # Corrupt everything from the origin onwards: the input must not change.
    corrupted = values.copy()
    corrupted[origin:] = 1e6
    x_past_corrupted, _ = build_input(corrupted, calendar, origin, 336, 24, wavelet)
    assert np.array_equal(x_past, x_past_corrupted)


def test_full_series_dwt_would_leak():
    """Documents the original bug: denoising the whole series lets future
    values change the 'past' inputs near the origin."""
    _, values, _ = synthetic_series()
    origin = 24 * 30
    corrupted = values.copy()
    corrupted[origin:] = 1e6
    whole_clean = denoise_windows(values, DB4)
    whole_corrupted = denoise_windows(corrupted, DB4)
    last_input_hours = slice(origin - 24, origin)
    assert not np.allclose(whole_clean[last_input_hours], whole_corrupted[last_input_hours])


def test_windows_are_aligned():
    index, values, calendar = synthetic_series()
    windows = build_windows(values, calendar, index, index[0], index[24 * 20 - 1],
                            window_hours=48, horizon_hours=24, stride_hours=1, wavelet=None)
    first = windows.origins[0]
    position = index.get_loc(first)
    assert position == 48                       # first sample needs 48 h of history
    assert np.allclose(windows.x_past[0, :, 0], values[0:48])
    assert np.allclose(windows.y[0], values[48:72])
    assert np.allclose(windows.x_future[0], calendar[48:72])
    # The last sample's target ends exactly at the requested last hour.
    last_position = index.get_loc(windows.origins[-1])
    assert last_position + 24 - 1 == 24 * 20 - 1


def test_windowed_dwt_matches_single_input():
    index, values, calendar = synthetic_series()
    windows = build_windows(values, calendar, index, index[0], index[24 * 20 - 1],
                            window_hours=336, horizon_hours=24, stride_hours=24, wavelet=DB4)
    position = index.get_loc(windows.origins[0])
    x_past, _ = build_input(values, calendar, position, 336, 24, DB4)
    assert np.allclose(windows.x_past[0], x_past)

def test_level_by_target_overrides_default():
    cfg = {"wavelet": "db4", "level": 3, "mode": "symmetric", "method": "approximation",
           "level_by_target": {"demand": 2}}
    assert WaveletConfig.from_config(cfg, "demand").level == 2
    assert WaveletConfig.from_config(cfg, "pv").level == 3      # not listed: default
    assert WaveletConfig.from_config(cfg).level == 3            # no target given: default
