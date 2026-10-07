"""Causal wavelet denoising: each input window is denoised on its own.

The original pipeline decomposed the whole year at once. A wavelet filter
reaches several samples to both sides, so every "denoised" hour then contained
information from hours after it, including the hours being predicted. Here the
transform only ever sees one input window, which ends at the forecast origin,
so no future value can enter an input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt

VALID_METHODS = ("approximation", "soft_threshold")


@dataclass(frozen=True)
class WaveletConfig:
    wavelet: str = "db4"
    level: int = 3
    mode: str = "symmetric"
    method: str = "approximation"
    @classmethod
    def from_config(cls, dwt_cfg: dict, target: str | None = None) -> "WaveletConfig":
        # A per-target level (chosen on validation) overrides the default level.
        level = int(dwt_cfg["level"])
        level_by_target = dwt_cfg.get("level_by_target") or {}
        if target is not None and target in level_by_target:
            level = int(level_by_target[target])
        return cls(
            wavelet=str(dwt_cfg["wavelet"]),
            level=level,
            mode=str(dwt_cfg["mode"]),
            method=str(dwt_cfg.get("method", "approximation")),
        )


def check_level(window_length: int, cfg: WaveletConfig) -> None:
    """Refuse decomposition levels too deep for the window length."""
    filter_length = pywt.Wavelet(cfg.wavelet).dec_len
    max_level = pywt.dwt_max_level(window_length, filter_length)
    if cfg.level < 1 or cfg.level > max_level:
        raise ValueError(
            f"Level {cfg.level} not possible for {cfg.wavelet} on {window_length} samples "
            f"(allowed 1..{max_level})"
        )


def denoise_windows(windows: np.ndarray, cfg: WaveletConfig) -> np.ndarray:
    """Denoise each row of ``windows`` (shape: n_windows x window_length) independently.

    approximation:  keep only the level-L approximation (all detail set to 0),
                    as in the original paper.
    soft_threshold: shrink detail coefficients with the universal threshold
                    sigma * sqrt(2 ln n), sigma estimated per window from the
                    finest detail level (Donoho & Johnstone).
    """
    if cfg.method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got '{cfg.method}'")

    windows = np.asarray(windows, dtype=float)
    single = windows.ndim == 1
    if single:
        windows = windows[np.newaxis, :]
    window_length = windows.shape[1]
    check_level(window_length, cfg)

    # axis=-1: transform along time, separately for every row (window).
    coeffs = pywt.wavedec(windows, cfg.wavelet, level=cfg.level, mode=cfg.mode, axis=-1)

    if cfg.method == "approximation":
        new_coeffs = [coeffs[0]]
        for detail in coeffs[1:]:
            new_coeffs.append(np.zeros_like(detail))
    else:
        finest = coeffs[-1]
        # Median absolute deviation of the finest details estimates the noise
        # level robustly (large genuine peaks barely move the median).
        sigma = np.median(np.abs(finest), axis=-1, keepdims=True) / 0.6745
        threshold = sigma * np.sqrt(2.0 * np.log(window_length))
        new_coeffs = [coeffs[0]]
        for detail in coeffs[1:]:
            new_coeffs.append(pywt.threshold(detail, threshold, mode="soft"))

    rebuilt = pywt.waverec(new_coeffs, cfg.wavelet, mode=cfg.mode, axis=-1)
    # Reconstruction can be one sample longer for odd lengths; trim to size.
    rebuilt = rebuilt[:, :window_length]
    return rebuilt[0] if single else rebuilt
