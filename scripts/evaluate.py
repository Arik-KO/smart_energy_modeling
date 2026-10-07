"""Summarize trained models: validation always, December only when confirmed.

Usage (from the project root):
    python scripts/evaluate.py --root runs/final                 # November (validation) only
    python scripts/evaluate.py --root runs/final --confirm-test  # also December (test)
    python scripts/evaluate.py --root runs/ablation --confirm-test

December is the held-out test month. Look at it once, after every modelling
decision has been made on November.
"""

from __future__ import annotations

import json
import shutil

import pandas as pd

from efs.cli import build_parser, start_run
from efs.config import resolve_path
from efs.data.dataset import load_dataset
from efs.evaluation.holdout import evaluate_folder, naive_rows, summarize
from efs.forecasting.final import find_model_folders

VALIDATION_COLUMNS = ["target", "arch", "strategy", "variant", "seed", "objective",
                      "hourly_mape", "daily_mape", "hourly_rmse", "daylight_nmae",
                      "skill_vs_naive", "best_epoch", "train_seconds"]


def collect_validation(folders) -> pd.DataFrame:
    rows = []
    for folder in folders:
        with (folder / "val_metrics.json").open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        row = {}
        for column in VALIDATION_COLUMNS:
            row[column] = metrics.get(column)
        rows.append(row)
    return pd.DataFrame(rows)


def log_table(logger, title: str, table: pd.DataFrame, columns: list[str]) -> None:
    present = [column for column in columns if column in table.columns]
    with pd.option_context("display.width", 200, "display.max_rows", 500,
                           "display.float_format", "{:.3f}".format):
        logger.info("%s\n%s", title, table[present].to_string(index=False))


def main() -> None:
    parser = build_parser("Summarize validation and (optionally) December test results")
    parser.add_argument("--root", default="runs/final", help="Folder containing trained models")
    parser.add_argument("--confirm-test", action="store_true",
                        help="Also forecast December (the held-out test month)")
    args = parser.parse_args()
    ctx = start_run(args, default_name="evaluate")
    logger = ctx.logger
    config = ctx.config

    root = resolve_path(config, args.root)
    folders = find_model_folders(root)
    if not folders:
        raise SystemExit(f"No trained models found under {root}")
    logger.info("Found %d trained models under %s", len(folders), root)

    # November: always safe to look at.
    validation = collect_validation(folders)
    validation.to_csv(ctx.run_dir / "val_all.csv", index=False, encoding="utf-8")
    group_keys = ["target", "arch", "strategy", "variant"]
    val_summary = validation.drop(columns=["seed"]).groupby(group_keys).agg(["mean", "std"])
    val_summary.columns = [f"{name}_{stat}" for name, stat in val_summary.columns]
    val_summary = val_summary.reset_index()
    val_summary.to_csv(ctx.run_dir / "val_summary.csv", index=False, encoding="utf-8")
    shutil.copy2(ctx.run_dir / "val_summary.csv", root / "val_summary.csv")
    log_table(logger, "Validation (November, month-ahead), mean over seeds:", val_summary,
              group_keys + ["objective_mean", "objective_std", "hourly_mape_mean",
                            "daily_mape_mean", "skill_vs_naive_mean"])

    if not args.confirm_test:
        logger.info("December not evaluated (add --confirm-test when all decisions are final)")
        return

    # December: the held-out test month.
    dataset = load_dataset(resolve_path(config, config["data"]["dataset_file"]))
    rows = naive_rows(dataset, config)
    for number, folder in enumerate(folders, start=1):
        logger.info("[%d/%d] December forecasts for %s", number, len(folders), folder.relative_to(root))
        rows.extend(evaluate_folder(folder, dataset, config, ctx.device))

    all_rows = pd.DataFrame(rows)
    all_rows.to_csv(ctx.run_dir / "test_all.csv", index=False, encoding="utf-8")
    summary = summarize(all_rows)
    summary.to_csv(ctx.run_dir / "test_summary.csv", index=False, encoding="utf-8")
    shutil.copy2(ctx.run_dir / "test_summary.csv", root / "test_summary.csv")
    log_table(logger, "December test, mean (std) over seeds:", summary,
              ["target", "horizon", "arch", "strategy", "variant", "n_seeds",
               "hourly_mape_mean", "hourly_mape_std", "daily_mape_mean",
               "daylight_nmae_mean", "skill_vs_naive_mean", "skill_vs_naive_std"])
    logger.info("Saved test_all.csv and test_summary.csv to %s (summary also in %s)",
                ctx.run_dir, root)


if __name__ == "__main__":
    main()
