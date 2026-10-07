"""Evaluate the frozen demand models on several test months (Dec 2019 - Feb 2020).

No model is retrained or re-tuned: the final models (trained on Jan-Oct 2019)
forecast each month from its first hour, with observed demand up to that
origin as history. January and February 2020 were never used for any
decision, so they are fully out-of-sample. December is included as a check:
its numbers must equal the main December evaluation.

Usage (from the project root):
    python scripts/evaluate_months.py
"""

from __future__ import annotations

import pandas as pd

from efs.cli import build_parser, start_run
from efs.config import resolve_path
from efs.data.dataset import load_dataset
from efs.evaluation.holdout import (
    metric_rows,
    model_forecasts,
    month_positions,
    naive_forecasts,
    summarize,
)
from efs.forecasting.final import find_model_folders, load_trained

TARGET = "demand"


def main() -> None:
    parser = build_parser("Evaluate frozen demand models on several test months")
    args = parser.parse_args()
    ctx = start_run(args, default_name="evaluate_months")
    logger = ctx.logger
    config = ctx.config
    ext_cfg = config["extra_months"]

    dataset = load_dataset(resolve_path(config, ext_cfg["dataset_file"]))
    column = config["forecasting"]["target_columns"][TARGET]
    threshold = float(config["evaluation"]["daylight_threshold_kwh"])
    final_root = resolve_path(config, config["final"]["output_dir"])
    folders = find_model_folders(final_root / TARGET)
    logger.info("Found %d demand models", len(folders))

    rows = []
    for month in ext_cfg["months"]:
        start, end, origins = month_positions(dataset.index, month["start"], month["end"])
        hours = dataset.index[start:end]
        actual = pd.Series(dataset[column].to_numpy(dtype=float)[start:end], index=hours)

        # Naive baselines for this month (same information as the models).
        raw_values = dataset[column].to_numpy(dtype=float)
        naive_raw = naive_forecasts(raw_values, start, end, origins)
        naive = {name: pd.Series(values, index=hours) for name, values in naive_raw.items()}
        naive_identity = {"target": TARGET, "arch": "seasonal_naive", "strategy": "-",
                          "variant": "-", "seed": 0}
        for row in metric_rows(actual, naive, naive, TARGET, threshold, naive_identity):
            row["month"] = month["name"]
            rows.append(row)

        for folder in folders:
            loaded = load_trained(folder, ctx.device)
            scaled = loaded.scaler.transform(dataset[column].to_numpy(dtype=float))
            calendar = dataset[loaded.feature_names].to_numpy(dtype=float)
            forecasts = {}
            for horizon, values in model_forecasts(loaded, scaled, calendar, start, end,
                                                   origins, ctx.device).items():
                forecasts[horizon] = pd.Series(loaded.scaler.inverse_transform(values), index=hours)

            frame = pd.DataFrame({"actual_kwh": actual})
            for horizon, series in forecasts.items():
                frame[f"{horizon}_kwh"] = series
            frame.to_csv(folder / f"forecast_{month['name']}.csv", encoding="utf-8",
                         date_format="%Y-%m-%d %H:%M")

            identity = {key: loaded.meta[key] for key in ("target", "arch", "strategy", "variant", "seed")}
            for row in metric_rows(actual, forecasts, naive, TARGET, threshold, identity):
                row["month"] = month["name"]
                rows.append(row)
        logger.info("%s done (%d hours, %d weekly origins)", month["name"], end - start, len(origins))

    out_dir = resolve_path(config, ext_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = pd.DataFrame(rows)
    all_rows.to_csv(out_dir / "months_all.csv", index=False, encoding="utf-8")

    # Per month: mean/std over seeds.
    per_month = []
    for month_name, group in all_rows.groupby("month", sort=False):
        summary = summarize(group.drop(columns=["month"]))
        summary.insert(0, "month", month_name)
        per_month.append(summary)
    per_month = pd.concat(per_month, ignore_index=True)
    per_month.to_csv(out_dir / "months_summary.csv", index=False, encoding="utf-8")

    # Pooled over months: every seed x month run counts once.
    pooled = summarize(all_rows.drop(columns=["month"]))
    pooled.to_csv(out_dir / "months_pooled.csv", index=False, encoding="utf-8")

    # Consistency check: December must reproduce the main evaluation.
    main_summary = resolve_path(config, config["final"]["output_dir"]) / "test_summary.csv"
    if main_summary.exists():
        reference = pd.read_csv(main_summary)
        keys = ["target", "arch", "strategy", "horizon"]
        december = per_month[per_month.month == ext_cfg["months"][0]["name"]]
        merged = december.merge(reference, on=keys, suffixes=("", "_main"))
        difference = (merged["hourly_mape_mean"] - merged["hourly_mape_mean_main"]).abs().max()
        logger.info("December check vs main evaluation: max |MAPE difference| = %.6f", difference)

    table = per_month[(per_month.strategy.isin(["dayblock_recursive", "-"])) &
                      (per_month.horizon == "month_ahead")]
    wide = table.pivot(index="arch", columns="month", values="hourly_mape_mean")
    pooled_month = pooled[(pooled.strategy.isin(["dayblock_recursive", "-"])) &
                          (pooled.horizon == "month_ahead")].set_index("arch")
    wide["pooled_mean"] = pooled_month["hourly_mape_mean"]
    wide["pooled_std"] = pooled_month["hourly_mape_std"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        logger.info("Month-ahead demand MAPE (%%), day-block, mean over seeds:\n%s", wide.to_string())
    logger.info("Saved months_all.csv, months_summary.csv, months_pooled.csv to %s", out_dir)


if __name__ == "__main__":
    main()
