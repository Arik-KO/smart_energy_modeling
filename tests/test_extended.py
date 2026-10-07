"""Tests for the extended demand dataset and the month bookkeeping."""

import numpy as np
import pandas as pd

from efs.data.extended import average_buildings, read_building
from efs.evaluation.holdout import month_positions


def write_building(path, start, hours, values):
    times = pd.date_range(start, periods=hours, freq="h")
    frame = pd.DataFrame({"date": times.strftime("%Y-%m-%d"), "hour": times.hour, "energy_kWh": values})
    frame.to_csv(path, index=False)


def test_read_building_keeps_first_duplicate_and_interpolates(tmp_path):
    path = tmp_path / "Residential_1.csv"
    write_building(path, "2019-01-01", 5, [1.0, np.nan, 3.0, 4.0, 5.0])
    # Append a repeated hour (as in the autumn DST change) with a different value.
    with path.open("a", encoding="utf-8") as handle:
        handle.write("2019-01-01,4,99.0\n")
    series = read_building(path, pd.Timestamp("2019-01-01"), pd.Timestamp("2019-01-01 04:00"))
    assert len(series) == 5
    assert series.iloc[1] == 2.0            # linear interpolation of the gap
    assert series.iloc[4] == 5.0            # first occurrence kept, not 99


def test_average_uses_only_available_buildings(tmp_path):
    write_building(tmp_path / "Residential_1.csv", "2019-01-01", 4, [2.0, 2.0, 2.0, 2.0])
    # The second building starts two hours later (like building 27 in March 2019).
    write_building(tmp_path / "Residential_2.csv", "2019-01-01 02:00", 2, [4.0, 4.0])
    average = average_buildings(
        [tmp_path / "Residential_1.csv", tmp_path / "Residential_2.csv"],
        pd.Timestamp("2019-01-01"), pd.Timestamp("2019-01-01 03:00"))
    assert np.allclose(average.to_numpy(), [2.0, 2.0, 3.0, 3.0])


def test_month_positions_weekly_origins():
    index = pd.date_range("2020-01-01", "2020-03-01", freq="h")
    start, end, origins = month_positions(index, "2020-02-01 00:00", "2020-02-29 23:00")
    assert end - start == 29 * 24                      # leap year February
    assert origins == [start + k * 168 for k in range(5)]
