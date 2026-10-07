"""Train one model and score it on the November month-ahead forecast.

December is NOT touched here; it is evaluated once, at the end, by a
separate script.

Usage (from the project root):
    python scripts/train.py --target demand --arch tnn --strategy dayblock_recursive
    python scripts/train.py --target pv --arch dwt_lstm --strategy hourly_recursive --seed 3
    python scripts/train.py --target demand --arch lstm --set training.max_epochs=5   # quick test
"""

from __future__ import annotations

from efs.cli import build_parser, start_run
from efs.config import resolve_path
from efs.data.dataset import load_dataset
from efs.forecasting.pipeline import save_result, train_and_validate


def main() -> None:
    parser = build_parser("Train one forecasting model and validate it month-ahead")
    parser.add_argument("--target", required=True, choices=["demand", "pv"])
    parser.add_argument("--arch", required=True, choices=["lstm", "dwt_lstm", "tnn", "dwt_tnn"])
    parser.add_argument("--strategy", default="dayblock_recursive",
                        choices=["hourly_recursive", "dayblock_recursive"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_name = args.run_name or f"train_{args.target}_{args.arch}_{args.strategy}_s{args.seed}"
    args.run_name = run_name
    ctx = start_run(args, default_name="train")
    config = ctx.config

    dataset = load_dataset(resolve_path(config, config["data"]["dataset_file"]))
    result = train_and_validate(dataset, config, args.target, args.arch, args.strategy,
                                args.seed, ctx.device)
    save_result(result, ctx.run_dir, extra={"target": args.target, "arch": args.arch,
                                            "strategy": args.strategy, "seed": args.seed})
    ctx.logger.info("Saved model, scaler, metrics and forecasts to %s", ctx.run_dir)


if __name__ == "__main__":
    main()
