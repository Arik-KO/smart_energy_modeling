"""Daily battery model and schedule simulator.

Sign convention (all quantities in kWh per day):
  * request > 0 asks to CHARGE: energy drawn from the building's bus.
  * request < 0 asks to DISCHARGE: energy delivered to the building's bus.
  * net = demand - pv + charge - discharge; net > 0 is imported, net < 0 exported.

A request is first reduced to what is physically feasible; the SAME feasible
amounts are then used for the state-of-charge update and for the grid energy,
so energy is conserved even when the battery is full or empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from efs.scheduling.tariff import Tariff


@dataclass(frozen=True)
class BatterySpec:
    capacity_kwh: float
    round_trip_efficiency: float = 0.95
    initial_soc_fraction: float = 0.5
    max_cycles_per_day: float = 1.0
    allow_grid_charging: bool = True

    @classmethod
    def from_config(cls, battery_cfg: dict, capacity_kwh: float) -> "BatterySpec":
        return cls(
            capacity_kwh=float(capacity_kwh),
            round_trip_efficiency=float(battery_cfg["round_trip_efficiency"]),
            initial_soc_fraction=float(battery_cfg["initial_soc_fraction"]),
            max_cycles_per_day=float(battery_cfg["max_cycles_per_day"]),
            allow_grid_charging=bool(battery_cfg["allow_grid_charging"]),
        )

    @property
    def one_way_efficiency(self) -> float:
        # Round trip = charge efficiency x discharge efficiency; we split the
        # losses evenly: 0.95 round trip -> sqrt(0.95) ~ 0.975 each way.
        return math.sqrt(self.round_trip_efficiency)

    @property
    def initial_soc_kwh(self) -> float:
        return self.initial_soc_fraction * self.capacity_kwh

    @property
    def max_daily_throughput_kwh(self) -> float:
        return self.max_cycles_per_day * self.capacity_kwh


def feasible_step(
    request_kwh: float,
    soc_kwh: float,
    demand_kwh: float,
    pv_kwh: float,
    battery: BatterySpec,
) -> tuple[float, float]:
    """Reduce one day's request to feasible (charge, discharge) amounts."""
    eta = battery.one_way_efficiency
    charge = 0.0
    discharge = 0.0

    if request_kwh > 0.0:
        room_kwh = battery.capacity_kwh - soc_kwh
        limits = [request_kwh, battery.max_daily_throughput_kwh, room_kwh / eta]
        if not battery.allow_grid_charging:
            # Only the day's PV surplus may be stored.
            limits.append(max(pv_kwh - demand_kwh, 0.0))
        charge = max(min(limits), 0.0)

    elif request_kwh < 0.0:
        limits = [
            -request_kwh,
            battery.max_daily_throughput_kwh,
            soc_kwh * eta,
            # Discharge only serves the building's own load; exporting stored
            # energy never pays under this tariff (bought/stored at >= the
            # export price and 5% is lost on the way).
            max(demand_kwh - pv_kwh, 0.0),
        ]
        discharge = max(min(limits), 0.0)

    return charge, discharge


@dataclass
class ScheduleResult:
    """Daily trajectories and costs of one simulated schedule."""

    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    soc_kwh: np.ndarray          # length n_days + 1 (start of each day, then end)
    net_kwh: np.ndarray
    import_cost: np.ndarray
    export_revenue: np.ndarray
    tier1_import_kwh: np.ndarray
    tier2_import_kwh: np.ndarray
    terminal_shortfall_kwh: float
    refill_cost: float

    @property
    def total_import_cost(self) -> float:
        return float(self.import_cost.sum())

    @property
    def total_export_revenue(self) -> float:
        return float(self.export_revenue.sum())

    @property
    def net_cost(self) -> float:
        """Import cost - export revenue + cost of restoring the initial charge."""
        return self.total_import_cost - self.total_export_revenue + self.refill_cost

    def summary(self) -> dict[str, float]:
        return {
            "import_kwh": float(np.maximum(self.net_kwh, 0.0).sum()),
            "export_kwh": float(np.maximum(-self.net_kwh, 0.0).sum()),
            "tier2_import_kwh": float(self.tier2_import_kwh.sum()),
            "tier2_days": int((self.tier2_import_kwh > 0).sum()),
            "import_cost": self.total_import_cost,
            "export_revenue": self.total_export_revenue,
            "refill_cost": self.refill_cost,
            "net_cost": self.net_cost,
            "charged_kwh": float(self.charge_kwh.sum()),
            "discharged_kwh": float(self.discharge_kwh.sum()),
            "final_soc_kwh": float(self.soc_kwh[-1]),
        }


def simulate(
    demand_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    requests_kwh: np.ndarray,
    battery: BatterySpec,
    tariff: Tariff,
    initial_soc_kwh: float | None = None,
) -> ScheduleResult:
    """Run a daily schedule through the battery and tariff.

    ``initial_soc_kwh`` overrides the battery's default starting charge; the
    week-ahead scheduler uses it to continue from where the last week ended.
    """
    demand_kwh = np.asarray(demand_kwh, dtype=float)
    pv_kwh = np.asarray(pv_kwh, dtype=float)
    requests_kwh = np.asarray(requests_kwh, dtype=float)
    n_days = len(demand_kwh)
    if len(pv_kwh) != n_days or len(requests_kwh) != n_days:
        raise ValueError("demand, pv and requests must have the same length")

    eta = battery.one_way_efficiency
    charge = np.zeros(n_days)
    discharge = np.zeros(n_days)
    soc = np.zeros(n_days + 1)
    soc[0] = battery.initial_soc_kwh if initial_soc_kwh is None else float(initial_soc_kwh)

    for day in range(n_days):
        day_charge, day_discharge = feasible_step(
            requests_kwh[day], soc[day], demand_kwh[day], pv_kwh[day], battery
        )
        charge[day] = day_charge
        discharge[day] = day_discharge
        next_soc = soc[day] + eta * day_charge - day_discharge / eta
        # Clamp only tiny floating-point overshoots; the limits above already
        # keep the true value inside [0, capacity].
        soc[day + 1] = min(max(next_soc, 0.0), battery.capacity_kwh)

    net = demand_kwh - pv_kwh + charge - discharge
    tier1, tier2 = tariff.split_import(net)

    # Terminal condition: a schedule that ends with less stored energy than it
    # started with has "borrowed" that energy. Charge it the cost of refilling
    # the difference from the grid at the Tier-1 price, so every scheme is
    # compared on the same footing.
    shortfall = max(soc[0] - soc[-1], 0.0)
    refill_cost = (shortfall / eta) * tariff.tier1_price if shortfall > 0 else 0.0

    return ScheduleResult(
        charge_kwh=charge,
        discharge_kwh=discharge,
        soc_kwh=soc,
        net_kwh=net,
        import_cost=tariff.import_cost(net),
        export_revenue=tariff.export_revenue(net),
        tier1_import_kwh=tier1,
        tier2_import_kwh=tier2,
        terminal_shortfall_kwh=shortfall,
        refill_cost=refill_cost,
    )


def self_consumption_requests(demand_kwh: np.ndarray, pv_kwh: np.ndarray) -> np.ndarray:
    """Non-scheduled rule: store any PV surplus, discharge to cover any deficit.

    Requesting (pv - demand) each day means: charge the surplus on sunny days,
    discharge the shortfall otherwise. The simulator applies the physical limits.
    """
    return np.asarray(pv_kwh, dtype=float) - np.asarray(demand_kwh, dtype=float)


def daily_totals(dataset: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    # Plain strings keep pandas' partial-date slicing: "2019-12-30" means the
    # whole day, while "2019-12-30 23:00" means that exact hour.
    period = dataset.loc[str(start):str(end), ["demand_kwh", "pv_kwh"]]
    daily = period.resample("D").sum()
    # Guard against partial days, which would bias the daily tariff.
    hours_per_day = period["demand_kwh"].resample("D").count()
    if (hours_per_day != 24).any():
        raise ValueError("daily_totals needs whole days (24 hourly values each)")
    return daily
