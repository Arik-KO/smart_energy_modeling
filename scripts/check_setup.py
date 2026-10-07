"""Check that the project skeleton, environment and data files are ready.

Usage (from the project root):
    python scripts/check_setup.py
    python scripts/check_setup.py --set device=cpu --log-level DEBUG
"""

from __future__ import annotations

from efs.cli import build_parser, start_run
from efs.config import resolve_path


def main() -> None:
    parser = build_parser("Verify config, logging, seeding, GPU and data files")
    args = parser.parse_args()
    ctx = start_run(args, default_name="check_setup")
    logger = ctx.logger

    # 1. A tiny computation on the selected device proves PyTorch can use it.
    import torch

    left = torch.randn(256, 256, device=ctx.device)
    right = torch.randn(256, 256, device=ctx.device)
    checksum = (left @ right).sum().item()
    logger.info("Matrix multiply on %s OK (checksum %.3f)", ctx.device, checksum)

    # 2. The same seed must give the same random numbers (reproducibility).
    from efs.utils.seed import set_seed

    set_seed(123, deterministic=True)
    first = torch.rand(3).tolist()
    set_seed(123, deterministic=True)
    second = torch.rand(3).tolist()
    if first == second:
        logger.info("Seeding is reproducible")
    else:
        logger.error("Seeding is NOT reproducible: %s vs %s", first, second)

    # 3. The processed data files must exist where the config says.
    all_found = True
    data_cfg = ctx.config["data"]
    solar_key = "ninja_file" if data_cfg["solar_source"] == "ninja" else "legacy_solar_file"
    for key in ("energy_file", solar_key):
        path = resolve_path(ctx.config, data_cfg[key])
        if path.exists():
            size_kb = path.stat().st_size / 1024
            logger.info("Found %s: %s (%.0f KB)", key, path, size_kb)
        else:
            logger.error("Missing %s: expected at %s", key, path)
            all_found = False

    if all_found:
        logger.info("Setup check passed")
    else:
        logger.error("Setup check failed: see messages above")


if __name__ == "__main__":
    main()
