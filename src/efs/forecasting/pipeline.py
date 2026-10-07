"""One call that trains a model and scores it on the November month-ahead
forecast. Shared by the training script and the tuner, so both measure
models in exactly the same way."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from efs.evaluation.metrics import forecast_report, skill_score
from efs.forecasting.data import PreparedData, prepare_target
from efs.forecasting.models import (
    LevelNormalized,
    base_architecture,
    build_model,
    count_parameters,
    uses_dwt,
)
from efs.forecasting.recursive import recursive_forecast, seasonal_naive_forecast
from efs.forecasting.train import TrainingConfig, TrainingHistory, train_model
from efs.preprocessing.wavelet import WaveletConfig
from efs.utils.seed import set_seed

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    model: nn.Module
    prepared: PreparedData
    params: dict
    history: TrainingHistory
    val_forecast: pd.Series      # kWh, month-ahead from the start of validation
    val_actual: pd.Series        # kWh
    val_report: dict
    objective: float


def month_ahead_validation(model: nn.Module, prepared: PreparedData, config: dict,
                           device: torch.device) -> tuple[pd.Series, pd.Series, dict]:
    """Recursive forecast of the whole validation month from its first hour."""
    splits = config["data"]["splits"]
    val_start = pd.Timestamp(splits["val_start"])
    val_end = pd.Timestamp(splits["val_end"])
    origin = int(prepared.index.get_loc(val_start))
    total_hours = int(prepared.index.get_loc(val_end)) - origin + 1

    scaled = recursive_forecast(model, prepared.scaled_values, prepared.calendar, origin,
                                total_hours, prepared.window_hours, prepared.horizon_hours,
                                prepared.wavelet, device)
    hours = prepared.index[origin:origin + total_hours]
    forecast = pd.Series(prepared.scaler.inverse_transform(scaled), index=hours)
    actual = pd.Series(prepared.scaler.inverse_transform(
        prepared.scaled_values[origin:origin + total_hours]), index=hours)

    threshold = float(config["evaluation"]["daylight_threshold_kwh"])
    report = forecast_report(actual, forecast, prepared.target, threshold)

    # Seasonal naive from the same origin: the bar any model must clear.
    naive_scaled = seasonal_naive_forecast(prepared.scaled_values, origin, total_hours)
    naive = pd.Series(prepared.scaler.inverse_transform(naive_scaled), index=hours)
    naive_report = forecast_report(actual, naive, prepared.target, threshold)
    report["naive_hourly_rmse"] = naive_report["hourly_rmse"]
    report["skill_vs_naive"] = skill_score(report["hourly_rmse"], naive_report["hourly_rmse"])
    return forecast, actual, report


def train_and_validate(
    dataset: pd.DataFrame,
    config: dict,
    target: str,
    architecture: str,
    strategy: str,
    seed: int,
    device: torch.device,
    params: dict | None = None,
    window_hours: int | None = None,
    wavelet: WaveletConfig | None = None,
    training_cfg: TrainingConfig | None = None,
    epoch_callback: Callable[[int, float], None] | None = None,
) -> TrainResult:
    """Prepare data, build and train one model, and score it on validation."""
    if window_hours is None:
        window_hours = int(config["forecasting"]["window_hours"])
    if uses_dwt(architecture):
        if wavelet is None:
            wavelet = WaveletConfig.from_config(config["dwt"], target)
    else:
        wavelet = None
    if params is None:
        params = dict(config["models"][base_architecture(architecture)])
    if training_cfg is None:
        training_cfg = TrainingConfig.from_config(config["training"])

    prepared = prepare_target(dataset, config, target, strategy, window_hours, wavelet)

    # Seed right before creating the model so its initial weights depend only
    # on this seed, not on anything that ran earlier in the process.
    set_seed(seed, bool(config["deterministic"]), bool(config["deterministic_attention"]))

    n_features = len(prepared.feature_names)
    level_normalization = bool(config["forecasting"]["level_normalization"])
    model = build_model(architecture, n_features, prepared.horizon_hours, params,
                        level_normalization)
    logger.info("%s | %s | %s | seed %d | %d trainable parameters | params %s",
                target, architecture, strategy, seed, count_parameters(model), params)

    history = train_model(model, prepared.train, prepared.val, training_cfg, device, seed,
                          epoch_callback)
    forecast, actual, report = month_ahead_validation(model, prepared, config, device)

    objective_name = config["evaluation"]["objective_metric"][target]
    objective = float(report[objective_name])
    logger.info("Validation month-ahead | %s = %.3f | hourly MAPE %.2f%% | daily MAPE %.2f%% | "
                "skill vs naive %.3f | best epoch %d | %.0f s",
                objective_name, objective, report["hourly_mape"], report["daily_mape"],
                report["skill_vs_naive"], history.best_epoch, history.seconds)

    return TrainResult(model=model, prepared=prepared, params=params, history=history,
                       val_forecast=forecast, val_actual=actual, val_report=report,
                       objective=objective)


def save_result(result: TrainResult, folder: Path, extra: dict | None = None) -> None:
    """Weights, scaler, metrics, forecasts and learning curve for one model."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    wavelet = result.prepared.wavelet
    torch.save({
        "state_dict": result.model.state_dict(),
        "params": result.params,
        "level_normalization": isinstance(result.model, LevelNormalized),
        "window_hours": result.prepared.window_hours,
        "horizon_hours": result.prepared.horizon_hours,
        "feature_names": result.prepared.feature_names,
        "wavelet": None if wavelet is None else asdict(wavelet),
        # target / arch / strategy / seed, so the model can be rebuilt later
        # without guessing from folder names.
        "meta": dict(extra or {}),
    }, folder / "model.pt")
    result.prepared.scaler.save(folder / "scaler.json")

    metrics = dict(result.val_report)
    metrics["objective"] = result.objective
    metrics["best_epoch"] = result.history.best_epoch
    metrics["train_seconds"] = result.history.seconds
    if extra:
        metrics.update(extra)
    with (folder / "val_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    frame = pd.DataFrame({"actual_kwh": result.val_actual, "forecast_kwh": result.val_forecast})
    frame.to_csv(folder / "val_forecast.csv", encoding="utf-8", date_format="%Y-%m-%d %H:%M")

    curve = pd.DataFrame({"epoch": np.arange(1, len(result.history.train_loss) + 1),
                          "train_loss": result.history.train_loss,
                          "val_loss": result.history.val_loss})
    curve.to_csv(folder / "learning_curve.csv", index=False, encoding="utf-8")
