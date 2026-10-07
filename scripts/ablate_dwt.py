"""DWT ablation: wavelet basis x decomposition level for the DWT models.

Uses each DWT model's tuned settings and changes only the wavelet and level,
so any difference comes from the DWT configuration alone. The no-DWT
reference is the plain LSTM/TNN from the final runs.

Usage (from the project root):
    python scripts/ablate_dwt.py
Interrupted? Run the same command again: finished models are skipped.
"""

from __future__ import annotations

import time

from efs.cli import build_parser, start_run
from efs.config import resolve_path
from efs.data.dataset import load_dataset
from efs.forecasting.final import load_tuned_settings, train_and_save
from efs.preprocessing.wavelet import WaveletConfig


def main() -> None:
    parser = build_parser("DWT ablation over wavelet basis and decomposition level")
    parser.add_argument("--target", choices=["demand", "pv", "all"], default="all")
    args = parser.parse_args()
    ctx = start_run(args, default_name="ablate_dwt")
    logger = ctx.logger
    config = ctx.config

    ablation_cfg = config["ablation"]
    root = resolve_path(config, ablation_cfg["output_dir"])
    dataset = load_dataset(resolve_path(config, config["data"]["dataset_file"]))
    targets = ["demand", "pv"] if args.target == "all" else [args.target]
    mode = config["dwt"]["mode"]
    method = ablation_cfg["method"]

    jobs = []
    for target in targets:
        for arch in ablation_cfg["architectures"]:
            for wavelet_name in ablation_cfg["wavelets"]:
                for level in ablation_cfg["levels"]:
                    for seed in ablation_cfg["seeds"]:
                        jobs.append((target, arch, wavelet_name, int(level), int(seed)))

    started = time.perf_counter()
    for number, (target, arch, wavelet_name, level, seed) in enumerate(jobs, start=1):
        wavelet = WaveletConfig(wavelet=wavelet_name, level=level, mode=mode, method=method)
        variant = f"{wavelet_name}_L{level}_{method}"
        logger.info("[%d/%d] %s | %s | %s | seed %d", number, len(jobs), target, arch, variant, seed)
        settings = load_tuned_settings(config, target, arch)
        train_and_save(dataset, config, root, target, arch, ablation_cfg["strategy"], variant,
                       seed, ctx.device, settings, wavelet=wavelet)
        logger.info("[%d/%d] done | %.1f min elapsed", number, len(jobs),
                    (time.perf_counter() - started) / 60)

    logger.info("All %d ablation models are in %s", len(jobs), root)


if __name__ == "__main__":
    main()
