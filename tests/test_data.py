"""Tests for the time alignment, features and splits."""

import numpy as np
import pandas as pd
import pytest

from efs.data.features import calendar_features
from efs.data.loading import clock_to_standard_time, full_year_index
from efs.data.splits import split_bounds, split_frame

TIMEZONE = "America/Vancouver"


def make_patched_clock_year(year: int = 2019) -> tuple[pd.Series, pd.Series]:
    """Mimic the processed demand file: wall-clock hours with the DST patch.

    Every row's value is its true UTC hour number, so after conversion we can
    check that each LST hour carries the value that belongs to it.
    """
    lst_hours = full_year_index(year)
    utc = lst_hours + pd.Timedelta(hours=8)
    clock = utc.tz_localize("UTC").tz_convert(TIMEZONE).tz_localize(None)
    values = pd.Series(np.arange(len(lst_hours), dtype=float))
    frame = pd.DataFrame({"clock": clock, "value": values})

    # Patch 1: keep only the first of the two Nov 3 01:00 rows.
    frame = frame.drop_duplicates(subset="clock", keep="first")
    # Patch 2: insert the non-existent Mar 10 02:00 row.
    inserted = pd.DataFrame({"clock": [pd.Timestamp(f"{year}-03-10 02:00")], "value": [-1.0]})
    frame = pd.concat([frame, inserted]).sort_values("clock").reset_index(drop=True)
    return frame["clock"], frame["value"]


def test_dst_patch_is_undone():
    clock, values = make_patched_clock_year()
    assert len(clock) == 8760  # same size as the real processed file

    lst, n_filled = clock_to_standard_time(clock, values, TIMEZONE, -8, 2019)

    assert len(lst) == 8760
    assert lst.index.is_unique and lst.index.is_monotonic_increasing
    assert n_filled == 1                      # only the dropped autumn hour
    assert (lst != -1.0).all()                # the fake spring row is gone
    # Away from the one interpolated hour, every value is exactly its hour number.
    expected = pd.Series(np.arange(8760, dtype=float), index=lst.index)
    assert np.allclose(lst.to_numpy(), expected.to_numpy())


def test_calendar_features():
    index = pd.date_range("2019-12-23", "2019-12-26 23:00", freq="h")
    features = calendar_features(index, "CA", "BC")
    assert features.loc["2019-12-25", "is_holiday"].eq(1).all()
    assert features.loc["2019-12-24", "is_holiday"].eq(0).all()
    # Dec 23 2019 is a Monday.
    assert features.loc["2019-12-23", "is_weekend"].eq(0).all()
    encoded = features[["hour_sin", "hour_cos"]].to_numpy()
    assert np.allclose((encoded ** 2).sum(axis=1), 1.0)  # points lie on the unit circle


def test_splits_sizes_and_order():
    cfg = {
        "train_start": "2019-01-01 00:00", "train_end": "2019-10-31 23:00",
        "val_start": "2019-11-01 00:00", "val_end": "2019-11-30 23:00",
        "test_start": "2019-12-01 00:00", "test_end": "2019-12-30 23:00",
    }
    frame = pd.DataFrame({"x": 0.0}, index=full_year_index(2019))
    parts = split_frame(frame, cfg)
    assert len(parts["train"]) == 7296
    assert len(parts["val"]) == 720
    assert len(parts["test"]) == 720

    overlapping = dict(cfg, val_start="2019-10-31 12:00")
    with pytest.raises(ValueError):
        split_bounds(overlapping)
