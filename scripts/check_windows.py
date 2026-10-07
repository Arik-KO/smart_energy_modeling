"""Build training/validation windows for every target and strategy, and run
the leakage check on the real data.

Usage (from the project root):
    python scripts/check_windows.py
    python scripts/check_windows.py --set forecasting.window_hours=672 dwt.level=2
"""

from __future__ import annotations

import time

import numpy as np

from efs.cli import build_parser, start_run
from efs.config import resolve_path
from efs.data.dataset import load_dataset
from efs.forecasting.data import prepare_target
from efs.forecasting.windowing import build_input
from efs.preprocessing.wavelet import WaveletConfig


def leakage_check(prepared, n_origins: int, rng: np.random.Generator) -> bool:
    """Corrupt all values from each origin onwards; inputs must stay identical."""
    values = prepared.scaled_values
    low = prepared.window_hours
    high = len(values) - prepared.horizon_hours
    for origin in rng.integers(low, high, size=n_origins):
        clean_input, _ = build_input(values, prepared.calendar, int(origin),
                                     prepared.window_hours, prepared.horizon_hours, prepared.wavelet)
        corrupted = values.copy()
        corrupted[origin:] = 1e6
        corrupt_input, _ = build_input(corrupted, prepared.calendar, int(origin),
                                       prepared.window_hours, prepared.horizon_hours, prepared.wavelet)
        if not np.array_equal(clean_input, corrupt_input):
            return False
    return True


def main() -> None:
    parser = build_parser("Build forecasting windows and check for leakage")
    args = parser.parse_args()
    ctx = start_run(args, default_name="check_windows")
    logger = ctx.logger
    config = ctx.config

    dataset = load_dataset(resolve_path(config, config["data"]["dataset_file"]))
    window_hours = int(config["forecasting"]["window_hours"])
    wavelet = WaveletConfig.from_config(config["dwt"])
    rng = np.random.default_rng(int(config["seed"]))

    all_passed = True
    for target in config["forecasting"]["targets"]:
        for strategy in config["forecasting"]["strategies"]:
            for use_dwt in (False, True):
                started = time.perf_counter()
                prepared = prepare_target(dataset, config, target, strategy, window_hours,
                                          wavelet if use_dwt else None)
                elapsed = time.perf_counter() - started
                size_mb = prepared.train.x_past.nbytes / 1e6
                logger.info("   x_past %s | x_future %s | y %s | %.0f MB | built in %.1f s",
                            prepared.train.x_past.shape, prepared.train.x_future.shape,
                            prepared.train.y.shape, size_mb, elapsed)
                passed = leakage_check(prepared, n_origins=50, rng=rng)
                all_passed = all_passed and passed
                logger.info("   leakage check: %s", "PASSED" if passed else "FAILED")

    if all_passed:
        logger.info("All leakage checks passed")
    else:
        logger.error("Leakage detected: do not train until this is fixed")


if __name__ == "__main__":
    main()
