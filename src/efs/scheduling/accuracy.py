"""How accurate must forecasts be for forecast-driven scheduling to pay off?

The actual December data are turned into synthetic "forecasts" with a known,
controlled error, and the same schedulers plan on them. Every plan is scored
on the actual data, exactly as in the main experiments. Two kinds of error:

  random  day-to-day scatter: forecast = actual x (1 + e), e ~ Normal(0, sigma)
  bias    a systematic level error: forecast = actual x factor

applied to demand only or to PV only, so their effects can be separated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from efs.scheduling.battery import BatterySpec, self_consumption_requests, simulate
from efs.scheduling.ga import GAConfig, optimize_schedule
from efs.scheduling.planning import rule_based_requests
from efs.scheduling.tariff import Tariff


def with_random_error(actual: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Multiply each day by (1 + e), e ~ N(0, sigma); energy cannot go negative."""
    actual = np.asarray(actual, dtype=float)
    noise = rng.normal(0.0, sigma, size=len(actual))
    return np.maximum(actual * (1.0 + noise), 0.0)


def with_bias(actual: np.ndarray, factor: float) -> np.ndarray:
    """Scale every day by the same factor (e.g. 2.5 = forecasting 2.5x too much)."""
    return np.asarray(actual, dtype=float) * float(factor)


@dataclass
class PlanScores:
    ga_savings: float
    rule_savings: float


def score_forecast(demand_fc: np.ndarray, pv_fc: np.ndarray, demand: np.ndarray, pv: np.ndarray,
                   battery: BatterySpec, tariff: Tariff, ga_cfg: GAConfig,
                   rng: np.random.Generator, lookahead_days: int) -> PlanScores:
    """Plan once (month ahead) on the forecast; return savings on the actual data."""
    reference = simulate(demand, pv, np.zeros(len(demand)), BatterySpec(capacity_kwh=0.0), tariff).net_cost
    soc0 = battery.initial_soc_kwh

    rule_plan = rule_based_requests(demand_fc, pv_fc, battery, tariff, soc0, lookahead_days)
    warm_start = [np.zeros(len(demand_fc)), self_consumption_requests(demand_fc, pv_fc), rule_plan]
    ga_plan, _ = optimize_schedule(demand_fc, pv_fc, battery, tariff, ga_cfg, rng,
                                   seed_individuals=warm_start)

    ga_cost = simulate(demand, pv, ga_plan, battery, tariff).net_cost
    rule_cost = simulate(demand, pv, rule_plan, battery, tariff).net_cost
    return PlanScores(ga_savings=reference - ga_cost, rule_savings=reference - rule_cost)


def error_coordinates(actual: np.ndarray, forecast: np.ndarray) -> tuple[float, float]:
    """Describe a real forecast in the same terms as the synthetic ones.

    Returns (level factor, random sigma): the factor is total forecast / total
    actual; sigma is the spread of the day-to-day relative error once that
    level error has been removed.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    factor = float(forecast.sum() / actual.sum())
    usable = actual > 0
    relative = forecast[usable] / (actual[usable] * factor) - 1.0
    return factor, float(np.std(relative, ddof=1))
