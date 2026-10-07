"""Final training with tuned settings, and reloading trained models.

Folder layout (one folder per trained model):

    <root>/<target>/<architecture>/<strategy>/<variant>/seed_<k>/

``variant`` is "tuned" for the main experiments and e.g. "db4_L2_approximation"
in the DWT ablation.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn

from efs.config import resolve_path
from efs.forecasting.models import build_model
from efs.forecasting.pipeline import TrainResult, save_result, train_and_validate
from efs.forecasting.train import TrainingConfig
from efs.preprocessing.scaling import MinMaxScaler1D
from efs.preprocessing.wavelet import WaveletConfig

logger = logging.getLogger(__name__)


def load_tuned_settings(config: dict, target: str, architecture: str) -> dict:
    path = resolve_path(config, config["tuning"]["output_dir"]) / f"{target}_{architecture}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No tuned settings for {target}/{architecture}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def model_folder(root: Path, target: str, architecture: str, strategy: str,
                 variant: str, seed: int) -> Path:
    return Path(root) / target / architecture / strategy / variant / f"seed_{seed}"


def is_trained(folder: Path) -> bool:
    """A folder counts as done only when its last file (the metrics) exists."""
    return (folder / "model.pt").exists() and (folder / "val_metrics.json").exists()


def train_with_settings(dataset: pd.DataFrame, config: dict, target: str, architecture: str,
                        strategy: str, seed: int, device: torch.device, settings: dict,
                        wavelet: WaveletConfig | None = None) -> TrainResult:
    """Train one model with tuned network, training and window settings."""
    training_cfg = dataclasses.replace(TrainingConfig.from_config(config["training"]),
                                       **settings.get("training", {}))
    return train_and_validate(
        dataset, config, target, architecture, strategy, seed, device,
        params=dict(settings["model_params"]),
        window_hours=int(settings["window_hours"]),
        wavelet=wavelet,
        training_cfg=training_cfg,
    )


def train_and_save(dataset: pd.DataFrame, config: dict, root: Path, target: str,
                   architecture: str, strategy: str, variant: str, seed: int,
                   device: torch.device, settings: dict,
                   wavelet: WaveletConfig | None = None) -> Path:
    """Train one model unless it already exists (so interrupted runs resume)."""
    folder = model_folder(root, target, architecture, strategy, variant, seed)
    if is_trained(folder):
        logger.info("Skipping %s (already trained)", folder)
        return folder

    result = train_with_settings(dataset, config, target, architecture, strategy, seed,
                                 device, settings, wavelet)
    meta = {"target": target, "arch": architecture, "strategy": strategy,
            "variant": variant, "seed": seed}
    save_result(result, folder, extra=meta)
    return folder


@dataclasses.dataclass
class LoadedModel:
    model: nn.Module
    scaler: MinMaxScaler1D
    window_hours: int
    horizon_hours: int
    feature_names: list[str]
    wavelet: WaveletConfig | None
    meta: dict


def load_trained(folder: Path, device: torch.device) -> LoadedModel:
    """Rebuild a saved model exactly as it was trained."""
    folder = Path(folder)
    checkpoint = torch.load(folder / "model.pt", map_location=device, weights_only=False)
    meta = checkpoint["meta"]
    model = build_model(meta["arch"], len(checkpoint["feature_names"]),
                        int(checkpoint["horizon_hours"]), checkpoint["params"],
                        bool(checkpoint["level_normalization"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    wavelet = None
    if checkpoint["wavelet"] is not None:
        wavelet = WaveletConfig(**checkpoint["wavelet"])

    return LoadedModel(
        model=model,
        scaler=MinMaxScaler1D.load(folder / "scaler.json"),
        window_hours=int(checkpoint["window_hours"]),
        horizon_hours=int(checkpoint["horizon_hours"]),
        feature_names=list(checkpoint["feature_names"]),
        wavelet=wavelet,
        meta=meta,
    )


def find_model_folders(root: Path) -> list[Path]:
    """Every trained model folder below ``root``, in a stable order."""
    folders = []
    for path in sorted(Path(root).rglob("model.pt")):
        folders.append(path.parent)
    return folders
