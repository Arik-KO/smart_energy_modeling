"""Tests for the value-of-forecast-accuracy helpers."""

import math

import numpy as np

from efs.scheduling.accuracy import error_coordinates, score_forecast, with_bias, with_random_error
from efs.scheduling.battery import BatterySpec, simulate
from efs.scheduling.ga import GAConfig
from efs.scheduling.lp import solve_perfect_information
from efs.scheduling.tariff import Tariff

TARIFF = Tariff(tier1_price=0.0941, tier2_price=0.1410, daily_threshold_kwh=22.1918,
                export_price=0.0941)
BATTERY = BatterySpec(capacity_kwh=20.0)


def test_perfect_forecast_captures_the_optimum():
    rng = np.random.default_rng(0)
    demand = rng.uniform(18, 40, size=30)
    pv = rng.uniform(0, 10, size=30)
    no_battery = simulate(demand, pv, np.zeros(30), BatterySpec(capacity_kwh=0.0), TARIFF).net_cost
    achievable = no_battery - solve_perfect_information(demand, pv, BATTERY, TARIFF).objective
    scores = score_forecast(demand, pv, demand, pv, BATTERY, TARIFF, GAConfig(), rng, 7)
    assert math.isclose(scores.ga_savings, achievable, rel_tol=1e-3)


def test_error_generators_and_coordinates():
    rng = np.random.default_rng(0)
    actual = np.full(1000, 10.0)
    assert np.allclose(with_random_error(actual, 0.0, rng), actual)
    noisy = with_random_error(actual, 0.1, rng)
    assert (noisy >= 0).all()
    factor, sigma = error_coordinates(actual, noisy)
    assert abs(factor - 1.0) < 0.02 and abs(sigma - 0.1) < 0.02
    factor, sigma = error_coordinates(actual, with_bias(actual, 2.5))
    assert math.isclose(factor, 2.5) and sigma < 1e-12
