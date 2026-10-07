"""Exact optimum of the daily battery schedule when demand and PV are known.

With the daily two-step tariff the cost is convex and piecewise linear in the
daily net import, so the whole scheduling problem is a linear programme (LP).
Solving it exactly gives:
  * the perfect-information benchmark (best possible savings), and
  * a yardstick for the GA: how close does the heuristic get to the optimum?

Variables per day d: charge c, discharge e, end-of-day charge s, Tier-1
import t1, Tier-2 import t2, export x. One extra variable r is the energy
shortfall at the end (charged the same refill cost as in ``simulate``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from efs.scheduling.battery import BatterySpec
from efs.scheduling.tariff import Tariff


@dataclass
class LPResult:
    requests_kwh: np.ndarray   # charge - discharge per day, ready for ``simulate``
    objective: float           # optimal net cost ($)


def solve_perfect_information(demand_kwh: np.ndarray, pv_kwh: np.ndarray,
                              battery: BatterySpec, tariff: Tariff,
                              initial_soc_kwh: float | None = None) -> LPResult:
    demand_kwh = np.asarray(demand_kwh, dtype=float)
    pv_kwh = np.asarray(pv_kwh, dtype=float)
    n = len(demand_kwh)
    eta = battery.one_way_efficiency
    soc0 = battery.initial_soc_kwh if initial_soc_kwh is None else float(initial_soc_kwh)
    net_load = demand_kwh - pv_kwh

    # Column positions of each variable block.
    c_idx = np.arange(0, n)
    e_idx = np.arange(n, 2 * n)
    s_idx = np.arange(2 * n, 3 * n)
    t1_idx = np.arange(3 * n, 4 * n)
    t2_idx = np.arange(4 * n, 5 * n)
    x_idx = np.arange(5 * n, 6 * n)
    r_idx = 6 * n
    n_vars = 6 * n + 1

    # Objective: Tier-1 and Tier-2 import costs, minus export credit,
    # plus the refill cost of any end-of-horizon shortfall.
    cost = np.zeros(n_vars)
    cost[t1_idx] = tariff.tier1_price
    cost[t2_idx] = tariff.tier2_price
    cost[x_idx] = -tariff.export_price
    cost[r_idx] = tariff.tier1_price / eta

    # Equalities: grid balance (n rows) and state-of-charge dynamics (n rows).
    a_eq = lil_matrix((2 * n, n_vars))
    b_eq = np.zeros(2 * n)
    for d in range(n):
        # t1 + t2 - x = (demand - pv) + charge - discharge
        a_eq[d, t1_idx[d]] = 1.0
        a_eq[d, t2_idx[d]] = 1.0
        a_eq[d, x_idx[d]] = -1.0
        a_eq[d, c_idx[d]] = -1.0
        a_eq[d, e_idx[d]] = 1.0
        b_eq[d] = net_load[d]

        # s_d = s_{d-1} + eta * c_d - e_d / eta   (s_{-1} is the initial charge)
        row = n + d
        a_eq[row, s_idx[d]] = 1.0
        a_eq[row, c_idx[d]] = -eta
        a_eq[row, e_idx[d]] = 1.0 / eta
        if d == 0:
            b_eq[row] = soc0
        else:
            a_eq[row, s_idx[d - 1]] = -1.0

    # Inequality: shortfall r >= soc0 - s_end   <=>   -s_end - r <= -soc0
    a_ub = lil_matrix((1, n_vars))
    a_ub[0, s_idx[-1]] = -1.0
    a_ub[0, r_idx] = -1.0
    b_ub = np.array([-soc0])

    # Bounds mirror the limits in ``feasible_step``.
    throughput = battery.max_daily_throughput_kwh
    bounds = [(0.0, 0.0)] * n_vars
    for d in range(n):
        charge_limit = throughput
        if not battery.allow_grid_charging:
            charge_limit = min(charge_limit, max(-net_load[d], 0.0))
        bounds[c_idx[d]] = (0.0, charge_limit)
        bounds[e_idx[d]] = (0.0, min(throughput, max(net_load[d], 0.0)))
        bounds[s_idx[d]] = (0.0, battery.capacity_kwh)
        bounds[t1_idx[d]] = (0.0, tariff.daily_threshold_kwh)
        bounds[t2_idx[d]] = (0.0, None)
        bounds[x_idx[d]] = (0.0, None)
    bounds[r_idx] = (0.0, None)

    result = linprog(cost, A_ub=a_ub.tocsr(), b_ub=b_ub, A_eq=a_eq.tocsr(), b_eq=b_eq,
                     bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"LP failed: {result.message}")

    charge = result.x[c_idx]
    discharge = result.x[e_idx]
    return LPResult(requests_kwh=charge - discharge, objective=float(result.fun))
