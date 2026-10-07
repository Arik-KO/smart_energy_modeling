"""December scheduling experiments.

For each battery capacity, schedules are planned from forecasts (or known
data) and ALWAYS scored on actual December demand and PV:

  no_battery      reference cost
  non_scheduled   store surplus / cover deficit, no forecasts (reactive)
  rule_based      [12]-style reserve rule, driven by forecasts
  ga              genetic algorithm, driven by forecasts
  perfect_ga      GA given the actual December data (what the GA can reach)
  perfect_lp      exact optimum given the actual December data (upper bound)

Forecast-driven schemes run for every architecture x seed, at two horizons:
month_ahead (one plan on Dec 1) and week_ahead (re-planned each week).

Usage (from the project root):
    python scripts/schedule.py
    python scripts/schedule.py --set scheduling.tariff.export_price=0.05 --run-name schedule_export005
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from efs.cli import build_parser, start_run
from efs.config import resolve_path
from efs.data.dataset import load_dataset
from efs.scheduling.battery import BatterySpec, daily_totals, self_consumption_requests, simulate
from efs.scheduling.forecasts import daily_forecasts
from efs.scheduling.ga import GAConfig, optimize_schedule
from efs.scheduling.lp import solve_perfect_information
from efs.scheduling.planning import rule_based_requests, run_rolling
from efs.scheduling.tariff import Tariff


def block_starts(days: pd.DatetimeIndex, origin_dates: list[str]) -> list[int]:
    """Day indices at which the week-ahead planner re-plans."""
    starts = []
    for date_text in origin_dates:
        starts.append(int(days.get_loc(pd.Timestamp(date_text))))
    return starts


def result_row(result, capacity, scheme, horizon, arch, seed, reference_cost) -> dict:
    row = {"capacity_kwh": capacity, "scheme": scheme, "horizon": horizon, "arch": arch, "seed": seed}
    row.update(result.summary())
    row["savings"] = reference_cost - result.net_cost
    return row


def trajectory_rows(result, days, capacity, scheme, horizon, arch, seed) -> list[dict]:
    rows = []
    for day_index, day in enumerate(days):
        rows.append({
            "capacity_kwh": capacity, "scheme": scheme, "horizon": horizon, "arch": arch,
            "seed": seed, "date": day.date().isoformat(),
            "charge_kwh": result.charge_kwh[day_index],
            "discharge_kwh": result.discharge_kwh[day_index],
            "soc_end_kwh": result.soc_kwh[day_index + 1],
            "net_import_kwh": result.net_kwh[day_index],
            "tier2_kwh": result.tier2_import_kwh[day_index],
        })
    return rows


def main() -> None:
    parser = build_parser("December scheduling experiments driven by the saved forecasts")
    args = parser.parse_args()
    ctx = start_run(args, default_name="schedule")
    logger = ctx.logger
    config = ctx.config
    sched_cfg = config["scheduling"]

    dataset = load_dataset(resolve_path(config, config["data"]["dataset_file"]))
    splits = config["data"]["splits"]
    actual = daily_totals(dataset, splits["test_start"], splits["test_end"])
    days = actual.index
    demand_actual = actual["demand_kwh"].to_numpy()
    pv_actual = actual["pv_kwh"].to_numpy()

    tariff = Tariff.from_config(sched_cfg["tariff"])
    ga_cfg = GAConfig.from_config(sched_cfg["ga"])
    ga_seed = int(sched_cfg["ga"]["seed"])
    lookahead = int(sched_cfg["rule_lookahead_days"])
    forecast_root = resolve_path(config, sched_cfg["forecast_root"])
    strategy = sched_cfg["forecast_strategy"]
    weekly_starts = block_starts(days, config["forecasting"]["horizons"]["week_ahead_origins"])
    horizons = {"month_ahead": [0], "week_ahead": weekly_starts}
    seeds = [int(seed) for seed in config["final"]["seeds"]]

    rows = []
    trajectories = []
    started = time.perf_counter()
    for capacity in sched_cfg["battery"]["capacities_kwh"]:
        battery = BatterySpec.from_config(sched_cfg["battery"], capacity)

        # References that need no forecasts.
        no_battery = simulate(demand_actual, pv_actual, np.zeros(len(days)),
                              BatterySpec(capacity_kwh=0.0), tariff)
        reference = no_battery.net_cost
        rows.append(result_row(no_battery, capacity, "no_battery", "-", "-", 0, reference))

        reactive = simulate(demand_actual, pv_actual,
                            self_consumption_requests(demand_actual, pv_actual), battery, tariff)
        rows.append(result_row(reactive, capacity, "non_scheduled", "-", "-", 0, reference))
        trajectories.extend(trajectory_rows(reactive, days, capacity, "non_scheduled", "-", "-", 0))

        lp = solve_perfect_information(demand_actual, pv_actual, battery, tariff)
        perfect_lp = simulate(demand_actual, pv_actual, lp.requests_kwh, battery, tariff)
        rows.append(result_row(perfect_lp, capacity, "perfect_lp", "actual", "-", 0, reference))
        trajectories.extend(trajectory_rows(perfect_lp, days, capacity, "perfect_lp", "actual", "-", 0))
        logger.info("%d kWh | LP optimum %.3f $ (re-simulated %.3f $) | no battery %.3f $",
                    capacity, lp.objective, perfect_lp.net_cost, reference)

        rng = np.random.default_rng([ga_seed, int(capacity)])
        # Warm start: the GA's first generation contains "do nothing", the
        # reactive rule and the reserve rule, so it refines the best simple
        # plan instead of searching from scratch.
        seeds_for_ga = [np.zeros(len(days)), self_consumption_requests(demand_actual, pv_actual),
                        rule_based_requests(demand_actual, pv_actual, battery, tariff,
                                            battery.initial_soc_kwh, lookahead)]
        perfect_requests, _ = optimize_schedule(demand_actual, pv_actual, battery, tariff, ga_cfg,
                                                rng, seed_individuals=seeds_for_ga)
        perfect_ga = simulate(demand_actual, pv_actual, perfect_requests, battery, tariff)
        rows.append(result_row(perfect_ga, capacity, "perfect_ga", "actual", "-", 0, reference))
        logger.info("%d kWh | GA with perfect information %.3f $ (optimality gap %.3f $)",
                    capacity, perfect_ga.net_cost, perfect_ga.net_cost - perfect_lp.net_cost)

        # Forecast-driven schemes.
        for arch in sched_cfg["forecast_architectures"]:
            for seed in seeds:
                demand_fc_all = daily_forecasts(forecast_root, "demand", arch, strategy, seed)
                pv_fc_all = daily_forecasts(forecast_root, "pv", arch, strategy, seed)
                for horizon, starts in horizons.items():
                    demand_fc = demand_fc_all[f"{horizon}_kwh"].to_numpy()
                    pv_fc = pv_fc_all[f"{horizon}_kwh"].to_numpy()
                    rng = np.random.default_rng([ga_seed, int(capacity), seed, len(starts)])

                    def ga_planner(d_fc, p_fc, soc, rng=rng):
                        seeds_block = [np.zeros(len(d_fc)), self_consumption_requests(d_fc, p_fc),
                                       rule_based_requests(d_fc, p_fc, battery, tariff, soc, lookahead)]
                        requests, _ = optimize_schedule(d_fc, p_fc, battery, tariff, ga_cfg, rng,
                                                        initial_soc_kwh=soc,
                                                        seed_individuals=seeds_block)
                        return requests

                    def rule_planner(d_fc, p_fc, soc):
                        return rule_based_requests(d_fc, p_fc, battery, tariff, soc, lookahead)

                    for scheme, planner in (("ga", ga_planner), ("rule_based", rule_planner)):
                        _, result = run_rolling(planner, demand_fc, pv_fc, demand_actual,
                                                pv_actual, starts, battery, tariff)
                        rows.append(result_row(result, capacity, scheme, horizon, arch, seed, reference))
                        trajectories.extend(trajectory_rows(result, days, capacity, scheme,
                                                            horizon, arch, seed))
            logger.info("%d kWh | %s done | %.1f min elapsed", capacity, arch,
                        (time.perf_counter() - started) / 60)

    table = pd.DataFrame(rows)
    # Capture ratio: share of the achievable (perfect-information) savings obtained.
    optimum = table[table.scheme == "perfect_lp"].set_index("capacity_kwh")["savings"]
    table["capture_ratio"] = table["savings"] / table["capacity_kwh"].map(optimum)
    table.to_csv(ctx.run_dir / "schedule_all.csv", index=False, encoding="utf-8")
    pd.DataFrame(trajectories).to_csv(ctx.run_dir / "schedule_trajectories.csv", index=False,
                                      encoding="utf-8")

    keys = ["capacity_kwh", "scheme", "horizon", "arch"]
    metrics = ["net_cost", "import_cost", "tier2_import_kwh", "savings", "capture_ratio"]
    summary = table.groupby(keys, sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(ctx.run_dir / "schedule_summary.csv", index=False, encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_rows", 500,
                           "display.float_format", "{:.3f}".format):
        logger.info("December scheduling, scored on actual data (mean/std over seeds):\n%s",
                    summary[keys + ["net_cost_mean", "savings_mean", "savings_std",
                                    "capture_ratio_mean", "tier2_import_kwh_mean"]].to_string(index=False))
    logger.info("Saved schedule_all.csv, schedule_summary.csv, schedule_trajectories.csv to %s",
                ctx.run_dir)


if __name__ == "__main__":
    main()
