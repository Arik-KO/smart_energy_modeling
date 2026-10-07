"""Configuration loading: a YAML file plus command-line overrides.

Overrides use dotted keys, e.g. ``training.batch_size=128``. The value part is
parsed with YAML, so ``128`` becomes an int, ``true`` a bool, ``[1, 2]`` a list,
and ``db4`` stays a string.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Key under which we store where the config came from. It is saved with the
# run so that every result can be traced back to its exact settings.
META_KEY = "_meta"


def find_project_root(start: Path) -> Path:
    """Walk upward from ``start`` until a folder containing pyproject.toml is found.

    This lets scripts resolve data paths correctly no matter which folder
    they are launched from. Falls back to the current working directory.
    """
    current = start.resolve()
    if current.is_file():
        current = current.parent

    candidates = [current]
    candidates.extend(current.parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ValueError(f"Top level of {path} must be a mapping, got {type(content).__name__}")
    return content


def parse_override(text: str) -> tuple[list[str], Any]:
    """Split ``"a.b.c=value"`` into (["a", "b", "c"], parsed_value)."""
    if "=" not in text:
        raise ValueError(f"Override '{text}' must look like key.subkey=value")

    # Split only on the first '=', so values may themselves contain '='.
    key_part, value_part = text.split("=", 1)
    keys = key_part.strip().split(".")
    for key in keys:
        if key == "":
            raise ValueError(f"Override '{text}' has an empty key segment")

    value = yaml.safe_load(value_part) if value_part.strip() != "" else None
    return keys, value


def apply_override(config: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set ``config[k1][k2]...[kn] = value`` in place.

    Unknown keys raise an error instead of silently creating new entries:
    a typo such as ``trainig.batch_size=64`` would otherwise be ignored and
    the run would quietly use the default value.
    """
    node = config
    walked = []
    for key in keys[:-1]:
        walked.append(key)
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"Unknown config section '{'.'.join(walked)}'")
        node = node[key]

    last_key = keys[-1]
    if last_key not in node:
        raise KeyError(f"Unknown config key '{'.'.join(keys)}'")
    node[last_key] = value


def load_config(config_path: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load the YAML config, apply overrides, and record where it came from."""
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = copy.deepcopy(load_yaml(config_path))

    applied = []
    for text in overrides or []:
        keys, value = parse_override(text)
        apply_override(config, keys, value)
        applied.append(text)

    config[META_KEY] = {
        "config_path": str(config_path),
        "project_root": str(find_project_root(config_path)),
        "overrides": applied,
    }
    return config


def resolve_path(config: dict[str, Any], path_value: str | Path) -> Path:
    """Turn a (possibly relative) path from the config into an absolute Path."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    project_root = Path(config[META_KEY]["project_root"])
    return (project_root / path).resolve()
