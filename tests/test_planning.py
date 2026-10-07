"""Tests for the LP optimum, the GA, the rule-based planner and the economics."""

import math

import numpy as np

from efs.scheduling.battery import BatterySpec, self_consumption_requests, simulate
from efs.scheduling.economics import annuity_factor, break_even_cost_per_kwh, net_present_value
from efs.scheduling.ga import GAConfig, optimize_schedule
from efs.scheduling.lp import solve_perfect_information
from efs.scheduling.planning import rule_based_requests, run_rolling
from efs.scheduling.tariff import Tariff

TARIFF = Tariff(tier1_price=0.0941, tier2_price=0.1410, daily_threshold_kwh=22.1918,
                export_price=0.0941)
BATTERY = BatterySpec(capacity_kwh=20.0, round_trip_efficiency=0.95)


def december_like(n_days: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    demand = rng.uniform(18, 40, size=n_days)
    pv = rng.uniform(0, 10, size=n_days)
    return demand, pv


def test_lp_objective_matches_simulation():
    demand, pv = december_like()
    lp = solve_perfect_information(demand, pv, BATTERY, TARIFF)
    replay = simulate(demand, pv, lp.requests_kwh, BATTERY, TARIFF)
    assert math.isclose(lp.objective, replay.net_cost, abs_tol=1e-4)


def test_lp_beats_or_equals_every_other_schedule():
    demand, pv = december_like()
    lp_cost = solve_perfect_information(demand, pv, BATTERY, TARIFF).objective
    no_battery = simulate(demand, pv, np.zeros(30), BatterySpec(capacity_kwh=0.0), TARIFF).net_cost
    reactive = simulate(demand, pv, self_consumption_requests(demand, pv), BATTERY, TARIFF).net_cost
    rng = np.random.default_rng(1)
    for _ in range(50):
        random_cost = simulate(demand, pv, rng.uniform(-20, 20, 30), BATTERY, TARIFF).net_cost
        assert lp_cost <= random_cost + 1e-6
    assert lp_cost <= reactive + 1e-6 and lp_cost <= no_battery + 1e-6


def test_ga_gets_close_to_the_optimum():
    demand, pv = december_like()
    lp_cost = solve_perfect_information(demand, pv, BATTERY, TARIFF).objective
    no_battery = simulate(demand, pv, np.zeros(30), BatterySpec(capacity_kwh=0.0), TARIFF).net_cost
    cfg = GAConfig(population_size=50, generations=100)
    _, ga_cost = optimize_schedule(demand, pv, BATTERY, TARIFF, cfg, np.random.default_rng(0),
                                   seed_individuals=[np.zeros(30)])
    # At least 80% of the achievable saving, and never better than the optimum.
    assert ga_cost >= lp_cost - 1e-6
    assert (no_battery - ga_cost) >= 0.8 * (no_battery - lp_cost)


def test_rule_based_shifts_energy_to_peak_days():
    # Day 0 ordinary (net 15), day 1 peak (net 30): the rule should buy on day 0
    # and discharge on day 1, cutting Tier-2 energy.
    demand, pv = np.array([15.0, 30.0]), np.array([0.0, 0.0])
    battery = BatterySpec(capacity_kwh=20.0, initial_soc_fraction=0.0)
    requests = rule_based_requests(demand, pv, battery, TARIFF, 0.0, lookahead_days=7)
    assert requests[0] > 0 and requests[1] < 0
    with_rule = simulate(demand, pv, requests, battery, TARIFF)
    without = simulate(demand, pv, np.zeros(2), BatterySpec(capacity_kwh=0.0), TARIFF)
    assert with_rule.tier2_import_kwh.sum() < without.tier2_import_kwh.sum()


def test_rolling_runner_carries_actual_charge_between_blocks():
    demand, pv = december_like(14)
    seen_socs = []

    def planner(d_fc, p_fc, soc):
        seen_socs.append(soc)
        return np.full(len(d_fc), 3.0)          # charge a little every day

    _, result = run_rolling(planner, demand, pv, demand, pv, [0, 7], BATTERY, TARIFF)
    assert seen_socs[0] == BATTERY.initial_soc_kwh
    # The second block starts from the charge actually reached after day 7.
    assert math.isclose(seen_socs[1], result.soc_kwh[7])


def test_economics_by_hand():
    assert math.isclose(annuity_factor(0.05, 10), 7.7217, rel_tol=1e-4)
    # 20 kWh at $500/kWh = $10,000; saving $30/yr, O&M $5/yr, 10 years at 5%.
    assert math.isclose(net_present_value(10_000, 30, 5, 0.05, 10), -10_000 + 25 * 7.7217, rel_tol=1e-4)
    assert math.isclose(break_even_cost_per_kwh(30, 5, 20, 0.05, 10), 25 * 7.7217 / 20, rel_tol=1e-4)
