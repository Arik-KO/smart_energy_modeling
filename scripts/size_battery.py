"""Battery sizing on Jan-Nov data: best-case savings, economics and emissions.

For each capacity and export price, the perfect-information LP gives the
largest saving any scheduler could achieve on the sizing period. Those
upper-bound savings are then annualized and valued: NPV, payback and the
break-even installed cost per kWh. Because they are upper bounds, a negative
NPV here means the battery cannot pay for itself under this tariff, whatever
the forecasting or scheduling method.

Usage (from the project root):
    python scripts/size_battery.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from efs.cli import build_parser, start_run
from efs.config import resolve_path
from efs.data.dataset import load_dataset
from efs.scheduling.battery import BatterySpec, daily_totals, self_consumption_requests, simulate
from efs.scheduling.economics import (
    break_even_cost_per_kwh,
    net_present_value,
    simple_payback_years,
)
from efs.scheduling.lp import solve_perfect_information
from efs.scheduling.tariff import Tariff


def grid_energy(result) -> tuple[float, float]:
    imports = float(np.maximum(result.net_kwh, 0.0).sum())
    exports = float(np.maximum(-result.net_kwh, 0.0).sum())
    return imports, exports


def main() -> None:
    parser = build_parser("Battery sizing, economics and emissions (Jan-Nov, perfect information)")
    args = parser.parse_args()
    ctx = start_run(args, default_name="size_battery")
    logger = ctx.logger
    config = ctx.config
    sched_cfg = config["scheduling"]
    econ = config["economics"]

    dataset = load_dataset(resolve_path(config, config["data"]["dataset_file"]))
    splits = config["data"]["splits"]
    period = daily_totals(dataset, splits["train_start"], splits["val_end"])
    demand = period["demand_kwh"].to_numpy()
    pv = period["pv_kwh"].to_numpy()
    n_days = len(period)
    annualize = 365.0 / n_days
    years = int(econ["lifetime_years"])
    rate = float(econ["discount_rate"])
    om = float(econ["om_per_year"])
    emission_factor = float(econ["emission_factor_kg_per_kwh"])

    rows = []
    for export_price in sched_cfg["sensitivity"]["export_prices"]:
        tariff_cfg = dict(sched_cfg["tariff"])
        tariff_cfg["export_price"] = float(export_price)
        tariff = Tariff.from_config(tariff_cfg)
        no_battery = simulate(demand, pv, np.zeros(n_days), BatterySpec(capacity_kwh=0.0), tariff)
        base_imports, base_exports = grid_energy(no_battery)

        for capacity in sched_cfg["battery"]["capacities_kwh"]:
            battery = BatterySpec.from_config(sched_cfg["battery"], capacity)
            reactive = simulate(demand, pv, self_consumption_requests(demand, pv), battery, tariff)
            lp = solve_perfect_information(demand, pv, battery, tariff)
            optimum = simulate(demand, pv, lp.requests_kwh, battery, tariff)
            imports, exports = grid_energy(optimum)
            annual_saving = (no_battery.net_cost - optimum.net_cost) * annualize
            # Net grid energy (imports - exports) rises by the battery's losses.
            extra_grid_kwh = ((imports - exports) - (base_imports - base_exports)) * annualize

            row = {
                "export_price": float(export_price),
                "capacity_kwh": capacity,
                "no_battery_cost": no_battery.net_cost,
                "non_scheduled_cost": reactive.net_cost,
                "optimal_cost": optimum.net_cost,
                "annual_saving_optimal": annual_saving,
                "annual_saving_non_scheduled": (no_battery.net_cost - reactive.net_cost) * annualize,
                "break_even_cost_per_kwh": break_even_cost_per_kwh(annual_saving, om, capacity, rate, years),
                "annual_extra_grid_kwh": extra_grid_kwh,
                "annual_extra_emissions_kg": extra_grid_kwh * emission_factor,
            }
            for cost_per_kwh in econ["battery_cost_per_kwh"]:
                capital = float(cost_per_kwh) * capacity
                row[f"npv_at_{int(cost_per_kwh)}"] = net_present_value(capital, annual_saving, om, rate, years)
                row[f"payback_at_{int(cost_per_kwh)}"] = simple_payback_years(capital, annual_saving, om)
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(ctx.run_dir / "sizing.csv", index=False, encoding="utf-8")
    shown = ["export_price", "capacity_kwh", "no_battery_cost", "optimal_cost",
             "annual_saving_optimal", "annual_saving_non_scheduled", "break_even_cost_per_kwh",
             "annual_extra_emissions_kg"]
    for cost_per_kwh in econ["battery_cost_per_kwh"]:
        shown.append(f"npv_at_{int(cost_per_kwh)}")
    with pd.option_context("display.width", 220, "display.float_format", "{:.2f}".format):
        logger.info("Sizing on %d days (Jan-Nov), savings annualized x%.3f, lifetime %d y, "
                    "discount %.1f%%:\n%s", n_days, annualize, years, 100 * rate,
                    table[shown].to_string(index=False))
    logger.info("Saved %s", ctx.run_dir / "sizing.csv")


if __name__ == "__main__":
    main()
