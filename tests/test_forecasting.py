"""Tests for the models, training loop, recursive forecaster and metrics."""

import numpy as np
import pandas as pd
import pytest
import torch

from efs.evaluation.metrics import regression_metrics, skill_score
from efs.forecasting.models import build_model, uses_dwt
from efs.forecasting.recursive import (
    one_step_forecast,
    recursive_forecast,
    rolling_origin_forecast,
    seasonal_naive_forecast,
)
from efs.forecasting.train import TrainingConfig, train_model
from efs.forecasting.windowing import build_windows
from efs.preprocessing.wavelet import WaveletConfig

SMALL = {
    "lstm": {"hidden_size": 16, "num_layers": 1, "dropout": 0.0, "mlp_units": 16, "mlp_dropout": 0.0},
    "tnn": {"d_model": 16, "num_heads": 2, "num_layers": 1, "ff_dim": 32, "dropout": 0.0,
            "mlp_units": 16, "mlp_dropout": 0.0},
}
CPU = torch.device("cpu")


def daily_cycle(n_days: int = 40):
    hours = np.arange(24 * n_days)
    index = pd.date_range("2019-01-01", periods=len(hours), freq="h")
    values = 0.5 + 0.4 * np.sin(2 * np.pi * hours / 24)
    calendar = np.column_stack([np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)])
    return index, values, calendar


@pytest.mark.parametrize("arch", ["lstm", "tnn", "dwt_lstm", "dwt_tnn"])
@pytest.mark.parametrize("horizon", [1, 24])
def test_model_output_shape(arch, horizon):
    base = arch.removeprefix("dwt_")
    model = build_model(arch, n_features=3, horizon_hours=horizon, params=SMALL[base])
    output = model(torch.randn(5, 48, 4), torch.randn(5, horizon, 3))
    assert output.shape == (5, horizon)


def test_dwt_prefix():
    assert uses_dwt("dwt_tnn") and not uses_dwt("tnn")


def test_training_learns_a_daily_cycle():
    torch.manual_seed(0)
    index, values, calendar = daily_cycle()
    train = build_windows(values, calendar, index, index[0], index[24 * 30 - 1], 48, 24, 6, None)
    val = build_windows(values, calendar, index, index[24 * 30], index[24 * 35 - 1], 48, 24, 6, None)
    model = build_model("lstm", 2, 24, SMALL["lstm"])
    cfg = TrainingConfig(batch_size=32, max_epochs=40, learning_rate=3e-3,
                         early_stopping_patience=40, log_every=100)
    history = train_model(model, train, val, cfg, CPU, seed=0)
    assert history.val_loss[history.best_epoch - 1] < history.val_loss[0] * 0.5


def test_recursive_forecast_never_reads_the_future():
    index, values, calendar = daily_cycle()
    model = build_model("tnn", 2, 24, SMALL["tnn"])
    origin = 24 * 30
    wavelet = WaveletConfig(wavelet="db4", level=2)
    clean = recursive_forecast(model, values, calendar, origin, 120, 96, 24, wavelet, CPU)
    corrupted = values.copy()
    corrupted[origin:] = 1e6
    again = recursive_forecast(model, corrupted, calendar, origin, 120, 96, 24, wavelet, CPU)
    assert np.allclose(clean, again)
    assert clean.shape == (120,)


def test_hourly_recursion_handles_partial_last_block():
    index, values, calendar = daily_cycle()
    model = build_model("lstm", 2, 24, SMALL["lstm"])
    # 50 hours with 24-hour blocks: 24 + 24 + 2.
    forecast = recursive_forecast(model, values, calendar, 24 * 30, 50, 48, 24, None, CPU)
    assert forecast.shape == (50,)


def test_rolling_origins_use_actuals_up_to_each_origin():
    index, values, calendar = daily_cycle()
    model = build_model("lstm", 2, 24, SMALL["lstm"])
    origins = [24 * 30, 24 * 32]
    rolled = rolling_origin_forecast(model, values, calendar, origins, 24 * 34, 48, 24, None, CPU)
    second = recursive_forecast(model, values, calendar, 24 * 32, 48, 48, 24, None, CPU)
    assert rolled.shape == (96,)
    assert np.allclose(rolled[48:], second)


def test_one_step_forecast_shape():
    index, values, calendar = daily_cycle()
    model = build_model("lstm", 2, 1, SMALL["lstm"])
    forecast = one_step_forecast(model, values, calendar, 24 * 30, 72, 48, None, CPU)
    assert forecast.shape == (72,) and (forecast >= 0).all()


def test_seasonal_naive_repeats_last_week():
    values = np.arange(400, dtype=float)
    forecast = seasonal_naive_forecast(values, 336, 200, season_hours=168)
    assert np.allclose(forecast[:168], values[168:336])
    assert np.allclose(forecast[168:], values[168:200])


def test_metrics_by_hand():
    actual = np.array([1.0, 2.0, 0.0, 4.0])
    predicted = np.array([1.5, 1.5, 0.5, 4.0])
    metrics = regression_metrics(actual, predicted)
    assert np.isclose(metrics["mae"], 0.375)                       # (0.5+0.5+0.5+0)/4
    assert np.isclose(metrics["rmse"], np.sqrt(0.75 / 4))
    # MAPE skips the zero actual: (0.5/1 + 0.5/2 + 0/4) / 3 = 25%
    assert np.isclose(metrics["mape"], 25.0) and metrics["mape_hours"] == 3
    assert np.isclose(skill_score(0.5, 1.0), 0.5)
@pytest.mark.parametrize("pooling", ["mean", "last", "mean_last"])
def test_transformer_pooling_modes(pooling):
    params = dict(SMALL["tnn"], pooling=pooling)
    model = build_model("tnn", 3, 24, params)
    assert model(torch.randn(4, 48, 4), torch.randn(4, 24, 3)).shape == (4, 24)


def test_last_token_pooling_sees_recent_hours():
    """Two windows with the same average but different recent hours must give
    different outputs once the last token is used (mean pooling barely reacts)."""
    torch.manual_seed(0)
    x_future = torch.zeros(1, 24, 2)
    smooth = torch.zeros(1, 48, 3)
    smooth[:, :, 0] = 0.5
    spiky = smooth.clone()
    spiky[:, -1, 0] += 1.0       # a spike in the most recent hour ...
    spiky[:, 0, 0] -= 1.0        # ... balanced by a dip at the start (same mean)

    def response(pooling):
        torch.manual_seed(0)
        model = build_model("tnn", 2, 24, dict(SMALL["tnn"], pooling=pooling))
        model.eval()
        with torch.no_grad():
            return (model(spiky, x_future) - model(smooth, x_future)).abs().mean().item()

    assert response("mean_last") > response("mean")

def test_level_normalization_adds_back_the_window_mean():
    from efs.forecasting.models import LevelNormalized

    class Zero(torch.nn.Module):
        def forward(self, x_past, x_future):
            # Record what the inner model receives, and predict "no deviation".
            self.seen = x_past
            return torch.zeros(x_past.shape[0], x_future.shape[1])

    inner = Zero()
    model = LevelNormalized(inner)
    x_past = torch.rand(3, 48, 4)
    output = model(x_past, torch.zeros(3, 24, 3))
    window_mean = x_past[:, :, 0].mean(dim=1, keepdim=True)
    assert torch.allclose(output, window_mean.expand(3, 24))
    # The inner model sees centred values and untouched calendar features.
    assert torch.allclose(inner.seen[:, :, 0].mean(dim=1), torch.zeros(3), atol=1e-6)
    assert torch.equal(inner.seen[:, :, 1:], x_past[:, :, 1:])
