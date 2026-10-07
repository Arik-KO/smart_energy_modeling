"""Forecast error metrics, always computed in physical units (kWh)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(actual: np.ndarray, predicted: np.ndarray,
                       mape_floor: float = 1e-6) -> dict[str, float]:
    """MAE, RMSE, R2, MAPE, sMAPE and nMAE for two equally long arrays.

    MAPE divides by the actual value, so it is undefined when the actual is
    zero (PV at night). Hours with |actual| <= mape_floor are left out of MAPE,
    and the number of hours used is reported so readers can judge it.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {predicted.shape}")

    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    total_variance = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1.0 - np.sum(error ** 2) / total_variance) if total_variance > 0 else float("nan")

    usable = np.abs(actual) > mape_floor
    if usable.any():
        mape = float(100.0 * np.mean(np.abs(error[usable]) / np.abs(actual[usable])))
    else:
        mape = float("nan")

    # sMAPE is bounded (0-200%) and defined whenever actual and prediction are
    # not both zero, which makes it usable for PV.
    denominator = np.abs(actual) + np.abs(predicted)
    both_nonzero = denominator > mape_floor
    smape = float(100.0 * np.mean(2.0 * np.abs(error[both_nonzero]) / denominator[both_nonzero]))

    mean_actual = float(np.mean(actual))
    nmae = float(100.0 * mae / mean_actual) if mean_actual > 0 else float("nan")

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "smape": smape,
            "nmae": nmae, "mape_hours": int(usable.sum()), "n_hours": int(len(actual))}


def forecast_report(actual: pd.Series, predicted: pd.Series, target: str,
                    daylight_threshold_kwh: float = 0.05) -> dict[str, float]:
    """Hourly metrics, daily-total metrics, and (for PV) daylight-only metrics."""
    report = {}
    for name, value in regression_metrics(actual.to_numpy(), predicted.to_numpy()).items():
        report[f"hourly_{name}"] = value

    daily_actual = actual.resample("D").sum()
    daily_predicted = predicted.resample("D").sum()
    for name, value in regression_metrics(daily_actual.to_numpy(), daily_predicted.to_numpy()).items():
        report[f"daily_{name}"] = value

    if target == "pv":
        # Daylight hours only: removes the many trivially-correct night zeros
        # that inflate R2 and make MAPE undefined.
        daylight = actual.to_numpy() > daylight_threshold_kwh
        for name, value in regression_metrics(actual.to_numpy()[daylight],
                                              predicted.to_numpy()[daylight]).items():
            report[f"daylight_{name}"] = value

    return report


def skill_score(model_rmse: float, reference_rmse: float) -> float:
    """1 - RMSE_model / RMSE_reference: > 0 means better than the reference."""
    return float(1.0 - model_rmse / reference_rmse) if reference_rmse > 0 else float("nan")


def error_by_lead_day(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """MAE and MAPE of daily totals for lead day 1, 2, ... (how accuracy decays)."""
    daily_actual = actual.resample("D").sum().to_numpy()
    daily_predicted = predicted.resample("D").sum().to_numpy()
    rows = []
    for lead_day in range(len(daily_actual)):
        error = abs(daily_predicted[lead_day] - daily_actual[lead_day])
        rows.append({
            "lead_day": lead_day + 1,
            "abs_error_kwh": float(error),
            "ape_percent": float(100.0 * error / daily_actual[lead_day]) if daily_actual[lead_day] > 0 else float("nan"),
        })
    return pd.DataFrame(rows)
