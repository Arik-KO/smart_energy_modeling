"""Read the December forecasts saved by the evaluation, as daily totals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def forecast_file(root: Path, target: str, architecture: str, strategy: str, seed: int) -> Path:
    return Path(root) / target / architecture / strategy / "tuned" / f"seed_{seed}" / "test_forecast.csv"


def daily_forecasts(root: Path, target: str, architecture: str, strategy: str, seed: int) -> pd.DataFrame:
    """Daily totals (kWh/day) of the actual values and each forecast horizon."""
    path = forecast_file(root, target, architecture, strategy, seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing December forecasts (run evaluate.py --confirm-test): {path}")
    hourly = pd.read_csv(path, encoding="utf-8", parse_dates=["timestamp"], index_col="timestamp")
    return hourly.resample("D").sum()
