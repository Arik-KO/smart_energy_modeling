"""Choosing the compute device from the config value."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

VALID_DEVICES = ("auto", "cuda", "cpu")


def resolve_device(name: str):
    """Return a ``torch.device`` for ``auto``, ``cuda`` or ``cpu``.

    ``auto`` uses the GPU when available. An explicit ``cuda`` request fails
    loudly if no GPU is visible, instead of silently training on the CPU.
    """
    import torch

    name = str(name).lower()
    if name not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got '{name}'")

    if name == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if name == "cuda":
        raise RuntimeError("device=cuda was requested but torch.cuda.is_available() is False")

    logger.warning("No GPU visible; falling back to CPU")
    return torch.device("cpu")
