"""Shared command-line handling and run start-up for every script.

Each script does:

    parser = build_parser("What this script does")
    parser.add_argument(...)            # script-specific options, if any
    args = parser.parse_args()
    ctx = start_run(args, default_name="train")

and then uses ``ctx.config``, ``ctx.run_dir``, ``ctx.logger`` and ``ctx.device``.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from efs.config import META_KEY, load_config, resolve_path
from efs.logging_utils import get_logger, setup_logging
from efs.utils.device import resolve_device
from efs.utils.run_dir import (
    collect_environment_info,
    create_run_dir,
    save_config_snapshot,
    save_json,
)
from efs.utils.seed import set_seed

DEFAULT_CONFIG = Path("configs") / "default.yaml"


@dataclass
class RunContext:
    """Everything a script needs after start-up."""

    config: dict[str, Any]
    run_dir: Path
    logger: logging.Logger
    device: Any  # torch.device; typed loosely so this module does not import torch


def build_parser(description: str) -> argparse.ArgumentParser:
    """Arguments shared by all scripts."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set seed=7 training.batch_size=128",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Suffix for the run folder name (defaults to the script's name)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (overrides logging.level in the config)",
    )
    return parser


def start_run(args: argparse.Namespace, default_name: str) -> RunContext:
    """Load config, create the run folder, set up logging, seed, and pick the device."""
    config = load_config(args.config, args.overrides)

    runs_root = resolve_path(config, config["project"]["runs_dir"])
    run_name = args.run_name or default_name
    run_dir = create_run_dir(runs_root, run_name)

    console_level = args.log_level or config["logging"]["level"]
    setup_logging(
        console_level=console_level,
        log_file=run_dir / "run.log",
        file_level=config["logging"]["file_level"],
    )
    logger = get_logger(f"scripts.{default_name}")

    # Seed before touching the GPU: the deterministic cuBLAS setting is only
    # read when CUDA initialises (see efs.utils.seed).
    set_seed(int(config["seed"]), bool(config["deterministic"]),
             bool(config["deterministic_attention"]))
    device = resolve_device(config["device"])

    save_config_snapshot(config, run_dir)
    project_root = Path(config[META_KEY]["project_root"])
    environment = collect_environment_info(project_root)
    save_json(environment, run_dir / "environment.json")

    logger.info("Run folder: %s", run_dir)
    logger.info("Config: %s", config[META_KEY]["config_path"])
    if config[META_KEY]["overrides"]:
        logger.info("Overrides: %s", " ".join(config[META_KEY]["overrides"]))
    logger.info("Device: %s | seed: %s", device, config["seed"])
    if environment.get("git_commit") is None:
        logger.warning("No git commit recorded; commit your code so results stay traceable")

    return RunContext(config=config, run_dir=run_dir, logger=logger, device=device)
