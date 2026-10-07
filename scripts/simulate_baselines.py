"""Simulate the two schedule-free baselines on actual data.

  * no battery
  * non-scheduled (self-consumption rule) for each battery capacity

Usage (from the project root):
    python scripts/simulate_baselines.py
    python scripts/simulate_baselines.py --set scheduling.tariff.export_price=0.05
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from efs.cli import build_parser, start_run
from efs.config import resolve_path
from efs.data.dataset import load_dataset
from efs.scheduling.battery import BatterySpec, daily_totals, self_consumption_requests, simulate
from efs.scheduling.tariff import Tariff


def main() -> None:
    parser = build_parser("Simulate no-battery and non-scheduled baselines on actual data")
    args = parser.parse_args()
    ctx = start_run(args, default_name="simulate_baselines")
    logger = ctx.logger
    config = ctx.config

    dataset = load_dataset(resolve_path(config, config["data"]["dataset_file"]))
    splits = config["data"]["splits"]
    periods = {
        "jan_nov": (splits["train_start"], splits["val_end"]),
        "december": (splits["test_start"], splits["test_end"]),
    }

    tariff = Tariff.from_config(config["scheduling"]["tariff"])
    battery_cfg = config["scheduling"]["battery"]

    rows = []
    for period_name, (start, end) in periods.items():
        daily = daily_totals(dataset, start, end)
        demand = daily["demand_kwh"].to_numpy()
        pv = daily["pv_kwh"].to_numpy()

        no_battery = simulate(demand, pv, np.zeros(len(demand)), BatterySpec(capacity_kwh=0.0), tariff)
        row = {"period": period_name, "scheme": "no_battery", "capacity_kwh": 0}
        row.update(no_battery.summary())
        rows.append(row)

        requests = self_consumption_requests(demand, pv)
        for capacity in battery_cfg["capacities_kwh"]:
            battery = BatterySpec.from_config(battery_cfg, capacity)
            result = simulate(demand, pv, requests, battery, tariff)
            row = {"period": period_name, "scheme": "non_scheduled", "capacity_kwh": capacity}
            row.update(result.summary())
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(ctx.run_dir / "baselines.csv", index=False, encoding="utf-8")

    columns = ["period", "scheme", "capacity_kwh", "import_kwh", "tier2_import_kwh",
               "tier2_days", "import_cost", "export_revenue", "net_cost"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.2f}".format):
        logger.info("Baselines on actual data:\n%s", table[columns].to_string(index=False))
    logger.info("Saved %s", ctx.run_dir / "baselines.csv")


if __name__ == "__main__":
    main()
