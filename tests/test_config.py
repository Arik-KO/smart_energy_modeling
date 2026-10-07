"""Tests for efs.config: override parsing, safety checks, and path resolution."""

from pathlib import Path

import pytest

from efs.config import (
    META_KEY,
    apply_override,
    find_project_root,
    load_config,
    parse_override,
    resolve_path,
)


def write_config(folder: Path) -> Path:
    """Create a tiny project with pyproject.toml and a config file."""
    (folder / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    config_dir = folder / "configs"
    config_dir.mkdir()
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        "seed: 1\n"
        "training:\n"
        "  batch_size: 32\n"
        "  learning_rate: 0.001\n"
        "data:\n"
        "  energy_file: data/energy.csv\n",
        encoding="utf-8",
    )
    return config_path


def test_parse_override_types():
    # YAML parsing turns text into the natural Python type.
    assert parse_override("a=3") == (["a"], 3)
    assert parse_override("a.b=true") == (["a", "b"], True)
    assert parse_override("a=[1, 2]") == (["a"], [1, 2])
    assert parse_override("a=db4") == (["a"], "db4")
    assert parse_override("a=1.0e-4") == (["a"], 1.0e-4)


def test_parse_override_rejects_bad_text():
    with pytest.raises(ValueError):
        parse_override("no_equals_sign")
    with pytest.raises(ValueError):
        parse_override("a..b=1")


def test_unknown_key_is_rejected():
    # A typo must fail loudly instead of silently using the default value.
    config = {"training": {"batch_size": 32}}
    with pytest.raises(KeyError):
        apply_override(config, ["trainig", "batch_size"], 64)
    with pytest.raises(KeyError):
        apply_override(config, ["training", "batch_sise"], 64)


def test_load_config_applies_overrides(tmp_path):
    config_path = write_config(tmp_path)
    config = load_config(config_path, ["training.batch_size=128", "seed=7"])
    assert config["training"]["batch_size"] == 128
    assert config["seed"] == 7
    assert config[META_KEY]["overrides"] == ["training.batch_size=128", "seed=7"]


def test_project_root_and_path_resolution(tmp_path):
    config_path = write_config(tmp_path)
    assert find_project_root(config_path) == tmp_path.resolve()

    config = load_config(config_path)
    resolved = resolve_path(config, config["data"]["energy_file"])
    assert resolved == (tmp_path / "data" / "energy.csv").resolve()
