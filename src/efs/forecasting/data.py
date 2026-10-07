"""Prepare scaled, windowed training and validation data for one target."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from efs.forecasting.windowing import WindowSet, build_windows
from efs.preprocessing.scaling import MinMaxScaler1D
from efs.preprocessing.wavelet import WaveletConfig

logger = logging.getLogger(__name__)


@dataclass
class PreparedData:
    target: str
    scaler: MinMaxScaler1D
    scaled_values: np.ndarray      # whole year, scaled with TRAIN statistics
    calendar: np.ndarray           # whole year, (n_hours, F)
    index: pd.DatetimeIndex
    feature_names: list[str]
    window_hours: int
    horizon_hours: int
    wavelet: WaveletConfig | None
    train: WindowSet
    val: WindowSet


def horizon_for_strategy(config: dict, strategy: str) -> int:
    """Hours predicted per model call: 1 (hour-by-hour) or 24 (day blocks)."""
    table = config["forecasting"]["strategy_output_hours"]
    if strategy not in table:
        raise ValueError(f"Unknown strategy '{strategy}'; expected one of {list(table)}")
    return int(table[strategy])


def prepare_target(
    dataset: pd.DataFrame,
    config: dict,
    target: str,
    strategy: str,
    window_hours: int,
    wavelet: WaveletConfig | None,
) -> PreparedData:
    """Scale with training statistics, then cut train and validation windows."""
    fc_cfg = config["forecasting"]
    splits = config["data"]["splits"]
    column = fc_cfg["target_columns"][target]
    feature_names = list(fc_cfg["calendar_features"][target])
    horizon_hours = horizon_for_strategy(config, strategy)

    train_start = pd.Timestamp(splits["train_start"])
    train_end = pd.Timestamp(splits["train_end"])
    val_start = pd.Timestamp(splits["val_start"])
    val_end = pd.Timestamp(splits["val_end"])

    # Fit the scaler on the training period ONLY, then apply it everywhere.
    raw = dataset[column].to_numpy(dtype=float)
    train_mask = (dataset.index >= train_start) & (dataset.index <= train_end)
    scaler = MinMaxScaler1D.fit(raw[train_mask])
    scaled = scaler.transform(raw)
    calendar = dataset[feature_names].to_numpy(dtype=float)

    stride = int(fc_cfg["stride_hours"])
    train = build_windows(scaled, calendar, dataset.index, train_start, train_end,
                          window_hours, horizon_hours, stride, wavelet)
    # Validation samples: every hour of November is predicted; inputs may look
    # back into October, which is in the past at those origins.
    val = build_windows(scaled, calendar, dataset.index, val_start, val_end,
                        window_hours, horizon_hours, stride, wavelet)

    logger.info(
        "%s | %s | L=%d H=%d | DWT=%s | train %d samples, val %d samples | "
        "scaler min %.3f max %.3f",
        target, strategy, window_hours, horizon_hours,
        "off" if wavelet is None else f"{wavelet.wavelet}/L{wavelet.level}/{wavelet.method}",
        len(train), len(val), scaler.minimum, scaler.maximum,
    )
    return PreparedData(
        target=target, scaler=scaler, scaled_values=scaled, calendar=calendar,
        index=dataset.index, feature_names=feature_names, window_hours=window_hours,
        horizon_hours=horizon_hours, wavelet=wavelet, train=train, val=val,
    )
