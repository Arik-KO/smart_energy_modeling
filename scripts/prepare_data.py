"""Build the modelling dataset (LST, demand + PV + calendar features) and validate it.

Usage (from the project root):
    python scripts/prepare_data.py
    python scripts/prepare_data.py --set data.solar_source=legacy data.pv_scale_factor=2.5
"""

from __future__ import annotations

import shutil

from efs.cli import build_parser, start_run
from efs.config import resolve_path
from efs.data.dataset import build_dataset, save_dataset, summarize_dataset, validate_dataset


def main() -> None:
    parser = build_parser("Build and validate the hourly LST dataset")
    args = parser.parse_args()
    ctx = start_run(args, default_name="prepare_data")
    logger = ctx.logger

    dataset = build_dataset(ctx.config)
    summarize_dataset(dataset, ctx.config)

    problems = validate_dataset(dataset, ctx.config)
    if problems:
        for problem in problems:
            logger.error("Validation: %s", problem)
        raise SystemExit("Dataset validation failed; nothing was saved")

    output_path = resolve_path(ctx.config, ctx.config["data"]["dataset_file"])
    save_dataset(dataset, output_path)
    # Keep a copy with the run, so the exact data behind a result is traceable.
    shutil.copy2(output_path, ctx.run_dir / output_path.name)
    logger.info("Saved %d rows x %d columns to %s", len(dataset), dataset.shape[1], output_path)


if __name__ == "__main__":
    main()
