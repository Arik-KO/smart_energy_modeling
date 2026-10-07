"""Hand-checked tests for the tariff and the daily battery simulator.

The worked example uses round-trip efficiency 0.81, so each direction is
exactly 0.9 and every number below can be verified with a calculator.
"""

import math

import numpy as np
import pytest

from efs.scheduling.battery import BatterySpec, self_consumption_requests, simulate
from efs.scheduling.tariff import Tariff

TARIFF = Tariff(tier1_price=0.1, tier2_price=0.2, daily_threshold_kwh=20.0, export_price=0.1)
BATTERY = BatterySpec(capacity_kwh=10.0, round_trip_efficiency=0.81,
                      initial_soc_fraction=0.5, max_cycles_per_day=1.0,
                      allow_grid_charging=True)


def test_tariff_two_steps_and_export():
    net = np.array([15.0, 25.0, -10.0])
    # 15 kWh all Tier 1; 25 kWh = 20 at Tier 1 + 5 at Tier 2; -10 = export.
    assert np.allclose(TARIFF.import_cost(net), [1.5, 20 * 0.1 + 5 * 0.2, 0.0])
    assert np.allclose(TARIFF.export_revenue(net), [0.0, 0.0, 1.0])


def test_worked_three_day_example():
    demand = np.array([25.0, 15.0, 30.0])
    pv = np.array([5.0, 10.0, 2.0])
    requests = np.array([-10.0, 8.0, -10.0])
    result = simulate(demand, pv, requests, BATTERY, TARIFF)

    # Day 1: discharge limited by stored energy: 5 kWh x 0.9 = 4.5 delivered.
    # Day 2: charge 8 from the grid; battery gains 8 x 0.9 = 7.2.
    # Day 3: discharge limited again: 7.2 x 0.9 = 6.48 delivered.
    assert np.allclose(result.discharge_kwh, [4.5, 0.0, 6.48])
    assert np.allclose(result.charge_kwh, [0.0, 8.0, 0.0])
    assert np.allclose(result.soc_kwh, [5.0, 0.0, 7.2, 0.0])

    # Net import: 25-5-4.5 = 15.5; 15-10+8 = 13; 30-2-6.48 = 21.52.
    assert np.allclose(result.net_kwh, [15.5, 13.0, 21.52])
    # Costs: 1.55; 1.30; 20 x 0.1 + 1.52 x 0.2 = 2.304.
    assert np.allclose(result.import_cost, [1.55, 1.30, 2.304])
    assert math.isclose(result.total_import_cost, 5.154)

    # Ended empty after starting at 5 kWh: refilling costs 5 / 0.9 x 0.1.
    assert math.isclose(result.terminal_shortfall_kwh, 5.0)
    assert math.isclose(result.refill_cost, 5.0 / 0.9 * 0.1)


def test_charging_stops_when_full():
    # Asking for 20 kWh when only 5 kWh of room is left: the battery takes
    # 5 / 0.9 = 5.56 kWh from the bus, and that (not 20) is what gets billed.
    result = simulate([10.0], [0.0], [20.0], BATTERY, TARIFF)
    assert math.isclose(result.charge_kwh[0], 5.0 / 0.9)
    assert math.isclose(result.soc_kwh[1], 10.0)
    assert math.isclose(result.net_kwh[0], 10.0 + 5.0 / 0.9)


def test_pv_only_charging():
    pv_only = BatterySpec(capacity_kwh=10.0, round_trip_efficiency=0.81,
                          initial_soc_fraction=0.0, allow_grid_charging=False)
    # Surplus of 3 kWh on day 1, none on day 2.
    result = simulate([7.0, 20.0], [10.0, 5.0], [8.0, 8.0], pv_only, TARIFF)
    assert np.allclose(result.charge_kwh, [3.0, 0.0])


def test_discharge_never_exceeds_load():
    full = BatterySpec(capacity_kwh=10.0, round_trip_efficiency=0.81, initial_soc_fraction=1.0)
    # Load 12 minus PV 10 leaves 2 kWh to cover: discharge is capped at 2.
    result = simulate([12.0], [10.0], [-10.0], full, TARIFF)
    assert math.isclose(result.discharge_kwh[0], 2.0)
    assert math.isclose(result.net_kwh[0], 0.0)


def test_no_battery_is_capacity_zero():
    result = simulate([25.0, 5.0], [0.0, 15.0], [5.0, -5.0],
                      BatterySpec(capacity_kwh=0.0), TARIFF)
    assert np.allclose(result.charge_kwh, 0.0)
    assert np.allclose(result.discharge_kwh, 0.0)
    assert np.allclose(result.net_kwh, [25.0, -10.0])


def test_energy_is_conserved_for_random_schedules():
    rng = np.random.default_rng(0)
    battery = BatterySpec(capacity_kwh=20.0, round_trip_efficiency=0.95)
    eta = battery.one_way_efficiency
    for _ in range(200):
        demand = rng.uniform(5, 40, size=30)
        pv = rng.uniform(0, 40, size=30)
        requests = rng.uniform(-30, 30, size=30)
        result = simulate(demand, pv, requests, battery, TARIFF)

        # Grid balance uses exactly the feasible amounts.
        assert np.allclose(result.net_kwh, demand - pv + result.charge_kwh - result.discharge_kwh)
        # Stored energy changes only through efficiency-weighted charge/discharge.
        change = result.soc_kwh[-1] - result.soc_kwh[0]
        expected = eta * result.charge_kwh.sum() - result.discharge_kwh.sum() / eta
        assert math.isclose(change, expected, abs_tol=1e-9)
        # Physical limits hold every day.
        assert (result.soc_kwh >= -1e-9).all() and (result.soc_kwh <= 20.0 + 1e-9).all()
        assert (result.charge_kwh * result.discharge_kwh == 0).all()


def test_self_consumption_rule():
    requests = self_consumption_requests([10.0, 30.0], [25.0, 5.0])
    assert np.allclose(requests, [15.0, -25.0])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        simulate([1.0, 2.0], [1.0], [0.0, 0.0], BATTERY, TARIFF)

def test_daily_totals_accepts_dates_and_timestamps():
    import pandas as pd

    from efs.scheduling.battery import daily_totals

    index = pd.date_range("2019-12-01 00:00", "2019-12-03 23:00", freq="h")
    frame = pd.DataFrame({"demand_kwh": 1.0, "pv_kwh": 0.5}, index=index)
    by_date = daily_totals(frame, "2019-12-01", "2019-12-02")
    by_hour = daily_totals(frame, "2019-12-01 00:00", "2019-12-02 23:00")
    assert len(by_date) == 2 and by_date.equals(by_hour)
    assert (by_date["demand_kwh"] == 24.0).all()
    with pytest.raises(ValueError):
        daily_totals(frame, "2019-12-01 06:00", "2019-12-02 23:00")  # partial first day
