"""Value of forecast accuracy for December scheduling.

Adds controlled errors to the actual December demand or PV, plans with the
GA and the rule-based scheduler, scores on the actual data, and reports the
capture ratio (share of the perfect-information saving obtained). The real
models' forecasts are described with the same error measures so they can be
placed on the curves.

Usage (from the project root):
    python scripts/value_of_accuracy.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from efs.cli import build_parser, start_run
from efs.config import resolve_path
from efs.data.dataset import load_dataset
from efs.scheduling.accuracy import error_coordinates, score_forecast, with_bias, with_random_error
from efs.scheduling.battery import BatterySpec, daily_totals, simulate
from efs.scheduling.forecasts import daily_forecasts
from efs.scheduling.ga import GAConfig
from efs.scheduling.lp import solve_perfect_information
from efs.scheduling.tariff import Tariff


def main() -> None:
    parser = build_parser("Value of forecast accuracy for forecast-driven scheduling")
    args = parser.parse_args()
    ctx = start_run(args, default_name="value_of_accuracy")
    logger = ctx.logger
    config = ctx.config
    sched_cfg = config["scheduling"]
    acc_cfg = config["accuracy_analysis"]

    dataset = load_dataset(resolve_path(config, config["data"]["dataset_file"]))
    splits = config["data"]["splits"]
    actual = daily_totals(dataset, splits["test_start"], splits["test_end"])
    demand = actual["demand_kwh"].to_numpy()
    pv = actual["pv_kwh"].to_numpy()

    tariff = Tariff.from_config(sched_cfg["tariff"])
    battery = BatterySpec.from_config(sched_cfg["battery"], acc_cfg["capacity_kwh"])
    ga_cfg = GAConfig.from_config(sched_cfg["ga"])
    lookahead = int(sched_cfg["rule_lookahead_days"])
    rng = np.random.default_rng(int(acc_cfg["seed"]))

    no_battery = simulate(demand, pv, np.zeros(len(demand)), BatterySpec(capacity_kwh=0.0), tariff).net_cost
    optimum = solve_perfect_information(demand, pv, battery, tariff).objective
    achievable = no_battery - optimum
    logger.info("Capacity %s kWh | achievable December saving %.3f $", acc_cfg["capacity_kwh"], achievable)

    def add_rows(rows, scores, **identity):
        for scheme, savings in (("ga", scores.ga_savings), ("rule_based", scores.rule_savings)):
            row = dict(identity)
            row.update({"scheme": scheme, "savings": savings, "capture_ratio": savings / achievable})
            rows.append(row)

    # 1. Random day-to-day error, on demand only or on PV only.
    random_rows = []
    for target in ("demand", "pv"):
        for sigma in acc_cfg["random_error_levels"]:
            for repetition in range(int(acc_cfg["repetitions"])):
                if target == "demand":
                    demand_fc, pv_fc = with_random_error(demand, sigma, rng), pv
                else:
                    demand_fc, pv_fc = demand, with_random_error(pv, sigma, rng)
                scores = score_forecast(demand_fc, pv_fc, demand, pv, battery, tariff, ga_cfg, rng, lookahead)
                add_rows(random_rows, scores, error_on=target, sigma=float(sigma), repetition=repetition)
        logger.info("Random-error curves for %s done", target)

    # 2. Systematic level error (bias), on demand only or on PV only.
    bias_rows = []
    for factor in acc_cfg["demand_bias_factors"]:
        scores = score_forecast(with_bias(demand, factor), pv, demand, pv, battery, tariff, ga_cfg, rng, lookahead)
        add_rows(bias_rows, scores, error_on="demand", factor=float(factor))
    for factor in acc_cfg["pv_bias_factors"]:
        scores = score_forecast(demand, with_bias(pv, factor), demand, pv, battery, tariff, ga_cfg, rng, lookahead)
        add_rows(bias_rows, scores, error_on="pv", factor=float(factor))

    # 3. Where the real models sit, in the same error terms.
    model_rows = []
    forecast_root = resolve_path(config, sched_cfg["forecast_root"])
    for arch in sched_cfg["forecast_architectures"]:
        for seed in config["final"]["seeds"]:
            demand_fc = daily_forecasts(forecast_root, "demand", arch, sched_cfg["forecast_strategy"], seed)["month_ahead_kwh"].to_numpy()
            pv_fc = daily_forecasts(forecast_root, "pv", arch, sched_cfg["forecast_strategy"], seed)["month_ahead_kwh"].to_numpy()
            demand_factor, demand_sigma = error_coordinates(demand, demand_fc)
            pv_factor, pv_sigma = error_coordinates(pv, pv_fc)
            model_rows.append({"arch": arch, "seed": int(seed),
                               "demand_factor": demand_factor, "demand_sigma": demand_sigma,
                               "pv_factor": pv_factor, "pv_sigma": pv_sigma})
    models = pd.DataFrame(model_rows)

    random_table = pd.DataFrame(random_rows)
    bias_table = pd.DataFrame(bias_rows)
    random_table.to_csv(ctx.run_dir / "accuracy_random.csv", index=False, encoding="utf-8")
    bias_table.to_csv(ctx.run_dir / "accuracy_bias.csv", index=False, encoding="utf-8")
    models.to_csv(ctx.run_dir / "accuracy_models.csv", index=False, encoding="utf-8")

    random_summary = (random_table.groupby(["error_on", "scheme", "sigma"])["capture_ratio"]
                      .agg(["mean", "std"]).reset_index())
    model_summary = models.groupby("arch")[["demand_factor", "demand_sigma", "pv_factor", "pv_sigma"]].mean()
    with pd.option_context("display.width", 200, "display.max_rows", 200,
                           "display.float_format", "{:.3f}".format):
        logger.info("Capture ratio vs random day-to-day error:\n%s", random_summary.to_string(index=False))
        logger.info("Capture ratio vs level bias (factor 1 = unbiased):\n%s",
                    bias_table[["error_on", "factor", "scheme", "capture_ratio"]].to_string(index=False))
        logger.info("Real month-ahead forecasts in the same terms (mean of seeds):\n%s",
                    model_summary.to_string())
    logger.info("Saved accuracy_random.csv, accuracy_bias.csv, accuracy_models.csv to %s", ctx.run_dir)


if __name__ == "__main__":
    main()
