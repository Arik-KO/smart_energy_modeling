"""December (test) evaluation of trained models at every horizon.

  month_ahead:     one recursive forecast from Dec 1 00:00 over 720 hours
  week_ahead:      a fresh recursive forecast from each weekly origin,
                   with actual data up to that origin
  one_hour_ahead:  1-hour-ahead with actual inputs (hourly models only),
                   the comparison with [12]

Each forecast is compared with the matching seasonal-naive baseline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from efs.evaluation.metrics import error_by_lead_day, forecast_report, skill_score
from efs.forecasting.final import LoadedModel, load_trained
from efs.forecasting.recursive import (
    one_step_forecast,
    recursive_forecast,
    rolling_origin_forecast,
    seasonal_naive_forecast,
)

logger = logging.getLogger(__name__)

KEY_METRICS = ("hourly_mae", "hourly_rmse", "hourly_mape", "hourly_r2", "hourly_nmae",
               "daily_mae", "daily_mape", "daily_rmse",
               "daylight_mae", "daylight_rmse", "daylight_nmae", "daylight_smape", "daylight_r2")


def period_positions(index: pd.DatetimeIndex, config: dict) -> tuple[int, int, list[int]]:
    """Start position, end position (exclusive) and week-ahead origin positions."""
    splits = config["data"]["splits"]
    start = int(index.get_loc(pd.Timestamp(splits["test_start"])))
    end = int(index.get_loc(pd.Timestamp(splits["test_end"]))) + 1
    origins = []
    for date_text in config["forecasting"]["horizons"]["week_ahead_origins"]:
        origins.append(int(index.get_loc(pd.Timestamp(date_text))))
    if origins[0] != start:
        raise ValueError("The first week-ahead origin must be the first test hour")
    return start, end, origins

def month_positions(index: pd.DatetimeIndex, start: str, end: str) -> tuple[int, int, list[int]]:
    """Start, end (exclusive) and weekly re-planning origins (every 7 days) for any test month."""
    first = int(index.get_loc(pd.Timestamp(start)))
    last = int(index.get_loc(pd.Timestamp(end))) + 1
    origins = list(range(first, last, 7 * 24))
    return first, last, origins

def naive_forecasts(values: np.ndarray, start: int, end: int, origins: list[int]) -> dict[str, np.ndarray]:
    """Seasonal-naive baselines for each horizon (same information as the models)."""
    total = end - start
    week_pieces = []
    boundaries = origins + [end]
    for origin, next_origin in zip(boundaries[:-1], boundaries[1:]):
        week_pieces.append(seasonal_naive_forecast(values, origin, next_origin - origin))
    return {
        "month_ahead": seasonal_naive_forecast(values, start, total),
        "week_ahead": np.concatenate(week_pieces),
        # 1 hour ahead, the natural naive forecast is "same as the last hour".
        "one_hour_ahead": np.asarray(values[start - 1:end - 1], dtype=float),
    }


def model_forecasts(loaded: LoadedModel, scaled: np.ndarray, calendar: np.ndarray,
                    start: int, end: int, origins: list[int],
                    device: torch.device) -> dict[str, np.ndarray]:
    """All horizons for one model, in scaled units."""
    forecasts = {
        "month_ahead": recursive_forecast(loaded.model, scaled, calendar, start, end - start,
                                          loaded.window_hours, loaded.horizon_hours,
                                          loaded.wavelet, device),
        "week_ahead": rolling_origin_forecast(loaded.model, scaled, calendar, origins, end,
                                              loaded.window_hours, loaded.horizon_hours,
                                              loaded.wavelet, device),
    }
    if loaded.horizon_hours == 1:
        forecasts["one_hour_ahead"] = one_step_forecast(loaded.model, scaled, calendar, start,
                                                        end - start, loaded.window_hours,
                                                        loaded.wavelet, device)
    return forecasts


def metric_rows(actual: pd.Series, forecasts: dict[str, pd.Series], naive: dict[str, pd.Series],
                target: str, daylight_threshold: float, identity: dict) -> list[dict]:
    rows = []
    for horizon, forecast in forecasts.items():
        report = forecast_report(actual, forecast, target, daylight_threshold)
        naive_report = forecast_report(actual, naive[horizon], target, daylight_threshold)
        row = dict(identity)
        row["horizon"] = horizon
        for key in KEY_METRICS:
            if key in report:
                row[key] = report[key]
        row["skill_vs_naive"] = skill_score(report["hourly_rmse"], naive_report["hourly_rmse"])
        rows.append(row)
    return rows


def evaluate_folder(folder: Path, dataset: pd.DataFrame, config: dict,
                    device: torch.device) -> list[dict]:
    """Forecast December with one saved model; save forecasts; return metric rows."""
    loaded = load_trained(folder, device)
    target = loaded.meta["target"]
    column = config["forecasting"]["target_columns"][target]

    scaled = loaded.scaler.transform(dataset[column].to_numpy(dtype=float))
    calendar = dataset[loaded.feature_names].to_numpy(dtype=float)
    start, end, origins = period_positions(dataset.index, config)
    hours = dataset.index[start:end]

    actual = pd.Series(dataset[column].to_numpy(dtype=float)[start:end], index=hours)
    forecasts = {}
    for horizon, values in model_forecasts(loaded, scaled, calendar, start, end, origins, device).items():
        forecasts[horizon] = pd.Series(loaded.scaler.inverse_transform(values), index=hours)
    naive = {}
    for horizon, values in naive_forecasts(scaled, start, end, origins).items():
        naive[horizon] = pd.Series(loaded.scaler.inverse_transform(values), index=hours)

    frame = pd.DataFrame({"actual_kwh": actual})
    for horizon, series in forecasts.items():
        frame[f"{horizon}_kwh"] = series
    frame.to_csv(folder / "test_forecast.csv", encoding="utf-8", date_format="%Y-%m-%d %H:%M")
    error_by_lead_day(actual, forecasts["month_ahead"]).to_csv(
        folder / "test_lead_days.csv", index=False, encoding="utf-8")

    identity = {key: loaded.meta[key] for key in ("target", "arch", "strategy", "variant", "seed")}
    threshold = float(config["evaluation"]["daylight_threshold_kwh"])
    return metric_rows(actual, forecasts, naive, target, threshold, identity)


def naive_rows(dataset: pd.DataFrame, config: dict) -> list[dict]:
    """Metric rows for the seasonal-naive baselines themselves."""
    start, end, origins = period_positions(dataset.index, config)
    hours = dataset.index[start:end]
    threshold = float(config["evaluation"]["daylight_threshold_kwh"])
    rows = []
    for target, column in config["forecasting"]["target_columns"].items():
        values = dataset[column].to_numpy(dtype=float)
        actual = pd.Series(values[start:end], index=hours)
        for horizon, forecast in naive_forecasts(values, start, end, origins).items():
            report = forecast_report(actual, pd.Series(forecast, index=hours), target, threshold)
            row = {"target": target, "arch": "seasonal_naive", "strategy": "-",
                   "variant": "-", "seed": 0, "horizon": horizon}
            for key in KEY_METRICS:
                if key in report:
                    row[key] = report[key]
            row["skill_vs_naive"] = 0.0
            rows.append(row)
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds for every model and horizon."""
    group_keys = ["target", "arch", "strategy", "variant", "horizon"]
    metric_columns = [column for column in rows.columns
                      if column not in group_keys and column != "seed"]
    grouped = rows.groupby(group_keys, sort=True)[metric_columns]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    counts = grouped.size().rename("n_seeds")
    return pd.concat([mean, std, counts], axis=1).reset_index()
