"""Build the demand-only dataset for Jan 2019 - Feb 2020 from the raw HUE files,
and check that its 2019 part matches the processed dataset used for training.

Usage (from the project root):
    python scripts/prepare_demand_extended.py
"""

from __future__ import annotations

import shutil

from efs.cli import build_parser, start_run
from efs.config import resolve_path
from efs.data.dataset import load_dataset, save_dataset
from efs.data.extended import build_extended_demand, compare_with_processed


def main() -> None:
    parser = build_parser("Build the extended (2019-2020) demand dataset from raw HUE files")
    args = parser.parse_args()
    ctx = start_run(args, default_name="prepare_demand_extended")
    logger = ctx.logger
    config = ctx.config

    def resolve(path_value):
        return resolve_path(config, path_value)

    extended = build_extended_demand(config, resolve)
    if extended.isna().any().any():
        raise SystemExit("The extended dataset has missing values; nothing was saved")

    processed = load_dataset(resolve(config["data"]["dataset_file"]))
    check = compare_with_processed(extended, processed)
    logger.info("2019 check vs processed dataset: %d hours compared, max difference %.6f kWh, "
                "%d hour(s) differ", check["hours_compared"], check["max_abs_difference_kwh"],
                check["hours_differing"])
    if check["max_abs_difference_kwh"] > 0.01:
        logger.warning("The rebuilt 2019 demand differs from the training data by more than "
                       "0.01 kWh somewhere; inspect before using the extra months")

    daily = extended["demand_kwh"].resample("MS").mean() * 24
    for month, value in daily.items():
        logger.info("%s  mean demand %.2f kWh/day", month.strftime("%Y-%m"), value)

    output = resolve(config["extra_months"]["dataset_file"])
    save_dataset(extended, output)
    shutil.copy2(output, ctx.run_dir / output.name)
    logger.info("Saved %d rows to %s", len(extended), output)


if __name__ == "__main__":
    main()
