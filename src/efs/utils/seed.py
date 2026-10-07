"""Seeding every random number generator used in the project."""

from __future__ import annotations

import logging
import os
import random
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True, deterministic_attention: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs).

    With ``deterministic=True`` the same seed gives the same result on the
    same machine, except (when ``deterministic_attention=False``) for tiny
    floating-point differences in the Transformer's attention gradients.
    It must be called before the first CUDA operation.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Imported here so that config/logging work even where torch is absent.
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS (GPU matrix multiplication) is only reproducible with this
        # workspace setting; it is read when CUDA initialises, hence "before
        # the first CUDA operation". setdefault keeps a user-provided value.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        # cuDNN normally benchmarks several algorithms and picks the fastest,
        # which can differ between runs. Fix the choice instead.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if deterministic_attention:
            # The plain (math) attention kernel is fully reproducible but stores
            # the whole L x L attention matrix: with long windows and many heads
            # it can exceed a 6 GB GPU.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        else:
            # The memory-efficient kernel needs far less memory; its backward
            # pass is not bit-for-bit reproducible. We accept that and report
            # mean +/- std over several seeds instead.
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            warnings.filterwarnings("ignore", message=".*Memory Efficient attention defaults to a non-deterministic.*")
            warnings.filterwarnings("ignore", message=".*Flash attention defaults to a non-deterministic.*")

        # warn_only=True: if some operation has no deterministic version,
        # log a warning rather than crash the run.
        torch.use_deterministic_algorithms(True, warn_only=True)

    logger.debug("Seed set to %d (deterministic=%s, deterministic_attention=%s)",
                 seed, deterministic, deterministic_attention)
