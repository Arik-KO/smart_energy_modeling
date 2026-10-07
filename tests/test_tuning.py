"""Tests for turning the config's search space into trial settings."""

import optuna

from efs.config import load_config
from efs.forecasting.tuning import sample_settings, suggest


def test_suggest_each_type():
    trial = optuna.trial.FixedTrial({"a": 64, "b": 2, "c": 0.001})
    assert suggest(trial, "a", {"type": "categorical", "choices": [32, 64]}) == 64
    assert suggest(trial, "b", {"type": "int", "low": 1, "high": 3}) == 2
    assert suggest(trial, "c", {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}) == 0.001


def test_sample_settings_splits_model_training_and_window():
    config = load_config("configs/default.yaml")
    fixed = {
        "window_hours": 168, "learning_rate": 5e-4, "weight_decay": 1e-5, "dropout": 0.2,
        "mlp_units": 64, "mlp_dropout": 0.1,
        "d_model": 32, "num_heads": 4, "num_layers": 2, "ff_dim": 64,
    }
    model_params, training, window = sample_settings(optuna.trial.FixedTrial(fixed), config, "dwt_tnn")
    assert window == 168
    assert training == {"learning_rate": 5e-4, "weight_decay": 1e-5}
    assert model_params["d_model"] == 32 and model_params["dropout"] == 0.2
    # Settings that are not tuned keep their config defaults.
    assert model_params["pooling"] == config["models"]["tnn"]["pooling"]
    assert "learning_rate" not in model_params and "window_hours" not in model_params
