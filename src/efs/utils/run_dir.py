"""Run folders: one folder per script execution, holding everything needed
to trace a result back to its code, settings and environment."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def create_run_dir(runs_root: Path, name: str) -> Path:
    """Create ``runs_root/<YYYYmmdd-HHMMSS>_<name>`` and return it.

    If two runs start within the same second, a numeric suffix keeps them apart
    rather than letting the second overwrite the first.
    """
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{stamp}_{name}"
    run_dir = runs_root / base_name

    suffix = 1
    while run_dir.exists():
        run_dir = runs_root / f"{base_name}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True)
    return run_dir


def save_config_snapshot(config: dict[str, Any], run_dir: Path) -> Path:
    """Write the fully resolved config (after overrides) into the run folder."""
    path = Path(run_dir) / "config_resolved.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return path


def get_git_commit(project_root: Path) -> str | None:
    """Return the current git commit hash, or None if unavailable.

    Git may be missing, or the folder may not be a repository yet; neither
    should stop an experiment, so errors are swallowed and None returned.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip()


def collect_environment_info(project_root: Path) -> dict[str, Any]:
    """Versions and hardware, saved with each run for the paper's reproducibility section."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": get_git_commit(project_root),
    }

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_build"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None

    for package_name in ("numpy", "pandas", "pywt", "optuna", "sklearn"):
        try:
            module = __import__(package_name)
            info[package_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[package_name] = None

    return info


def save_json(data: dict[str, Any], path: Path) -> None:
    """Write a dictionary as pretty-printed UTF-8 JSON."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)
