"""Rule-based scheduler in the style of [12], and the rolling-horizon runner.

Every scheduler here is a *planner*: given forecasts for a block of days and
the battery's charge at the start of the block, it returns one request per
day. The runner executes the plan on ACTUAL data, carries the resulting
charge into the next block, and finally scores the whole month on actuals.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from efs.scheduling.battery import BatterySpec, feasible_step, simulate, ScheduleResult
from efs.scheduling.tariff import Tariff

Planner = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def rule_based_requests(demand_fc: np.ndarray, pv_fc: np.ndarray, battery: BatterySpec,
                        tariff: Tariff, initial_soc_kwh: float, lookahead_days: int) -> np.ndarray:
    """Reserve-based rule following the logic of [12] (Section III-A).

    For each forecast day:
      * surplus day (PV > demand): store the surplus;
      * peak day (net import above the threshold): discharge the part above
        the threshold, so the day is billed at Tier 1 only;
      * ordinary day: charge from the grid, within the remaining Tier-1
        headroom, until the stored energy covers the forecast Tier-2 excess
        of the next ``lookahead_days`` days (the ESmin reserve of [12]).
    The predicted charge is tracked on the forecasts to decide how much to buy.
    """
    demand_fc = np.asarray(demand_fc, dtype=float)
    pv_fc = np.asarray(pv_fc, dtype=float)
    n_days = len(demand_fc)
    eta = battery.one_way_efficiency
    threshold = tariff.daily_threshold_kwh
    net = demand_fc - pv_fc
    excess = np.maximum(net - threshold, 0.0)

    requests = np.zeros(n_days)
    soc = float(initial_soc_kwh)
    for day in range(n_days):
        if net[day] < 0.0:
            request = -net[day]
        elif net[day] > threshold:
            request = -(net[day] - threshold)
        else:
            upcoming = float(excess[day + 1:day + 1 + lookahead_days].sum())
            # Energy that must be stored now to deliver `upcoming` later.
            reserve = min(upcoming / eta, battery.capacity_kwh)
            buy = max(reserve - soc, 0.0) / eta
            request = min(buy, threshold - net[day])
        requests[day] = request

        # Track the charge the plan expects, using the forecasts.
        charge, discharge = feasible_step(request, soc, demand_fc[day], pv_fc[day], battery)
        soc = min(max(soc + eta * charge - discharge / eta, 0.0), battery.capacity_kwh)
    return requests


def run_rolling(planner: Planner, demand_fc: np.ndarray, pv_fc: np.ndarray,
                demand_actual: np.ndarray, pv_actual: np.ndarray, block_starts: list[int],
                battery: BatterySpec, tariff: Tariff) -> tuple[np.ndarray, ScheduleResult]:
    """Plan block by block on forecasts; execute on actuals; score the month.

    ``block_starts`` are day indices, e.g. [0] for one month-ahead plan or
    [0, 7, 14, 21, 28] for weekly re-planning. At each block start the
    planner receives the battery charge actually reached so far.
    """
    n_days = len(demand_actual)
    boundaries = list(block_starts) + [n_days]
    executed = np.zeros(n_days)
    soc = battery.initial_soc_kwh

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        executed[start:end] = planner(demand_fc[start:end], pv_fc[start:end], soc)
        # Replay everything executed so far on the actual data to find the
        # true state of charge at the start of the next block.
        so_far = simulate(demand_actual[:end], pv_actual[:end], executed[:end], battery, tariff)
        soc = float(so_far.soc_kwh[-1])

    return executed, simulate(demand_actual, pv_actual, executed, battery, tariff)
