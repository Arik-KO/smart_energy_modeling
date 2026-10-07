"""Tune hyperparameters with an identical Optuna budget for every model.

Usage (from the project root):
    python scripts/tune.py                                  # all targets x all architectures
    python scripts/tune.py --target demand --arch tnn       # one pair
    python scripts/tune.py --set tuning.n_trials=3 tuning.max_epochs=5   # quick test

Interrupted? Run the same command again: finished trials are kept and only
the remaining ones run.
"""

from __future__ import annotations

import optuna

from efs.cli import build_parser, start_run
from efs.config import resolve_path
from efs.data.dataset import load_dataset
from efs.forecasting.tuning import best_settings, run_study, save_best_settings

ALL_TARGETS = ["demand", "pv"]
ALL_ARCHS = ["lstm", "dwt_lstm", "tnn", "dwt_tnn"]


def main() -> None:
    parser = build_parser("Tune forecasting hyperparameters with Optuna")
    parser.add_argument("--target", choices=ALL_TARGETS + ["all"], default="all")
    parser.add_argument("--arch", choices=ALL_ARCHS + ["all"], default="all")
    args = parser.parse_args()
    ctx = start_run(args, default_name="tune")
    logger = ctx.logger
    config = ctx.config

    # Optuna's own per-trial messages would duplicate ours.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    targets = ALL_TARGETS if args.target == "all" else [args.target]
    archs = ALL_ARCHS if args.arch == "all" else [args.arch]
    dataset = load_dataset(resolve_path(config, config["data"]["dataset_file"]))

    summary_rows = []
    for target in targets:
        for arch in archs:
            study = run_study(dataset, config, target, arch, ctx.device)
            study.trials_dataframe().to_csv(ctx.run_dir / f"trials_{study.study_name}.csv",
                                            index=False, encoding="utf-8")
            settings = best_settings(study, config, target, arch)
            path = save_best_settings(settings, config)
            logger.info("%s best: %s = %.3f (trial %d) -> %s", study.study_name,
                        settings["objective_metric"], settings["objective_value"],
                        settings["best_trial"], path)
            summary_rows.append(f"{target:<7} {arch:<9} {settings['objective_metric']:<14} "
                                f"{settings['objective_value']:8.3f}  window {settings['window_hours']}")

    logger.info("Tuning summary:\n%s", "\n".join(summary_rows))


if __name__ == "__main__":
    main()
