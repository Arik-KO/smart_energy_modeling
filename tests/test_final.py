"""Tests for reloading trained models and for the test-period bookkeeping."""

import numpy as np
import pandas as pd
import torch

from efs.evaluation.holdout import naive_forecasts, summarize, period_positions
from efs.forecasting.final import find_model_folders, load_trained
from efs.forecasting.models import build_model
from efs.preprocessing.scaling import MinMaxScaler1D


def test_saved_model_reloads_with_identical_outputs(tmp_path):
    params = {"d_model": 16, "num_heads": 2, "num_layers": 1, "ff_dim": 32, "dropout": 0.0,
              "mlp_units": 16, "mlp_dropout": 0.0, "pooling": "mean_last"}
    model = build_model("dwt_tnn", 2, 24, params)
    model.eval()
    folder = tmp_path / "demand" / "dwt_tnn" / "dayblock_recursive" / "tuned" / "seed_0"
    folder.mkdir(parents=True)
    torch.save({
        "state_dict": model.state_dict(), "params": params, "level_normalization": False,
        "window_hours": 48, "horizon_hours": 24, "feature_names": ["a", "b"],
        "wavelet": {"wavelet": "db4", "level": 2, "mode": "symmetric", "method": "approximation"},
        "meta": {"target": "demand", "arch": "dwt_tnn", "strategy": "dayblock_recursive",
                 "variant": "tuned", "seed": 0},
    }, folder / "model.pt")
    MinMaxScaler1D(minimum=0.1, maximum=3.0).save(folder / "scaler.json")

    loaded = load_trained(folder, torch.device("cpu"))
    x_past, x_future = torch.randn(3, 48, 3), torch.randn(3, 24, 2)
    with torch.no_grad():
        assert torch.allclose(model(x_past, x_future), loaded.model(x_past, x_future))
    assert loaded.wavelet.level == 2 and loaded.scaler.maximum == 3.0
    assert find_model_folders(tmp_path) == [folder]


def test_positions_and_naive_baselines():
    index = pd.date_range("2019-11-01", "2019-12-31 23:00", freq="h")
    config = {"data": {"splits": {"test_start": "2019-12-01 00:00", "test_end": "2019-12-30 23:00"}},
              "forecasting": {"horizons": {"week_ahead_origins": ["2019-12-01", "2019-12-08"]}}}
    start, end, origins = period_positions(index, config)
    assert end - start == 720 and origins == [start, start + 168]

    values = np.arange(len(index), dtype=float)
    naive = naive_forecasts(values, start, end, origins)
    for forecast in naive.values():
        assert forecast.shape == (720,)
    # Week-ahead naive restarts from the actual week before each origin.
    assert naive["week_ahead"][168] == values[start + 168 - 168]
    # One hour ahead: the previous hour.
    assert naive["one_hour_ahead"][0] == values[start - 1]


def test_summary_mean_and_std():
    rows = pd.DataFrame({
        "target": ["demand"] * 3, "arch": ["tnn"] * 3, "strategy": ["s"] * 3,
        "variant": ["tuned"] * 3, "horizon": ["month_ahead"] * 3, "seed": [0, 1, 2],
        "hourly_mape": [30.0, 32.0, 34.0],
    })
    summary = summarize(rows)
    assert summary.loc[0, "hourly_mape_mean"] == 32.0
    assert summary.loc[0, "hourly_mape_std"] == 2.0
    assert summary.loc[0, "n_seeds"] == 3
