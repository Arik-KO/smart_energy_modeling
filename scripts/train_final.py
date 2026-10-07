"""Train the final models: tuned settings x strategies x seeds.

Usage (from the project root):
    python scripts/train_final.py                         # everything (80 models)
    python scripts/train_final.py --target demand --arch tnn
Interrupted? Run the same command again: finished models are skipped.
"""

from __future__ import annotations

import time

from efs.cli import build_parser, start_run
from efs.config import resolve_path
from efs.data.dataset import load_dataset
from efs.forecasting.final import load_tuned_settings, train_and_save

ALL_TARGETS = ["demand", "pv"]
ALL_ARCHS = ["lstm", "dwt_lstm", "tnn", "dwt_tnn"]


def main() -> None:
    parser = build_parser("Train final models with tuned settings and several seeds")
    parser.add_argument("--target", choices=ALL_TARGETS + ["all"], default="all")
    parser.add_argument("--arch", choices=ALL_ARCHS + ["all"], default="all")
    args = parser.parse_args()
    ctx = start_run(args, default_name="train_final")
    logger = ctx.logger
    config = ctx.config

    final_cfg = config["final"]
    root = resolve_path(config, final_cfg["output_dir"])
    dataset = load_dataset(resolve_path(config, config["data"]["dataset_file"]))
    targets = ALL_TARGETS if args.target == "all" else [args.target]
    archs = ALL_ARCHS if args.arch == "all" else [args.arch]

    jobs = []
    for target in targets:
        for arch in archs:
            for strategy in final_cfg["strategies"]:
                for seed in final_cfg["seeds"]:
                    jobs.append((target, arch, strategy, int(seed)))

    started = time.perf_counter()
    for number, (target, arch, strategy, seed) in enumerate(jobs, start=1):
        logger.info("[%d/%d] %s | %s | %s | seed %d", number, len(jobs), target, arch, strategy, seed)
        settings = load_tuned_settings(config, target, arch)
        train_and_save(dataset, config, root, target, arch, strategy, "tuned", seed,
                       ctx.device, settings)
        elapsed_minutes = (time.perf_counter() - started) / 60
        logger.info("[%d/%d] done | %.1f min elapsed", number, len(jobs), elapsed_minutes)

    logger.info("All %d final models are in %s", len(jobs), root)


if __name__ == "__main__":
    main()
