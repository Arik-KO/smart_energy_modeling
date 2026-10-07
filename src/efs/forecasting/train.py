"""Training loop with early stopping on the validation windows."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from efs.forecasting.windowing import WindowSet

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    batch_size: int = 128
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0
    log_every: int = 5

    @classmethod
    def from_config(cls, training_cfg: dict) -> "TrainingConfig":
        return cls(
            batch_size=int(training_cfg["batch_size"]),
            max_epochs=int(training_cfg["max_epochs"]),
            learning_rate=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg["weight_decay"]),
            early_stopping_patience=int(training_cfg["early_stopping_patience"]),
            gradient_clip=float(training_cfg["gradient_clip"]),
            log_every=int(training_cfg["log_every"]),
        )


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    seconds: float = 0.0


def to_tensor_dataset(windows: WindowSet) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(windows.x_past),
        torch.from_numpy(windows.x_future),
        torch.from_numpy(windows.y),
    )


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Mean squared error over a data loader (scaled units)."""
    model.eval()
    loss_function = nn.MSELoss(reduction="sum")
    total = 0.0
    count = 0
    for x_past, x_future, y in loader:
        x_past = x_past.to(device)
        x_future = x_future.to(device)
        y = y.to(device)
        total += loss_function(model(x_past, x_future), y).item()
        count += y.numel()
    return total / count


def train_model(
    model: nn.Module,
    train_windows: WindowSet,
    val_windows: WindowSet,
    cfg: TrainingConfig,
    device: torch.device,
    seed: int,
    epoch_callback: Callable[[int, float], None] | None = None,
) -> TrainingHistory:
    """Train with AdamW + MSE; keep the weights of the best validation epoch.

    ``epoch_callback(epoch, val_loss)`` is called after every epoch; the tuner
    uses it to stop hopeless trials early (it may raise an exception).
    """
    model.to(device)

    # A seeded generator makes the shuffling order reproducible.
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(to_tensor_dataset(train_windows), batch_size=cfg.batch_size,
                              shuffle=True, generator=generator)
    val_loader = DataLoader(to_tensor_dataset(val_windows), batch_size=cfg.batch_size * 4,
                            shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    loss_function = nn.MSELoss()
    history = TrainingHistory()
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for x_past, x_future, y in train_loader:
            x_past = x_past.to(device)
            x_future = x_future.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss = loss_function(model(x_past, x_future), y)
            loss.backward()
            # Clipping stops rare huge gradients from wrecking the weights,
            # a known risk for recurrent networks on long sequences.
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            running += loss.item()
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        val_loss = evaluate_loss(model, val_loader, device)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % cfg.log_every == 0:
            logger.info("epoch %3d | train %.5f | val %.5f | best %.5f (epoch %d)",
                        epoch, train_loss, val_loss, history.best_val_loss, history.best_epoch)

        if epoch_callback is not None:
            epoch_callback(epoch, val_loss)

        if not np.isfinite(train_loss):
            logger.warning("Training loss is not finite; stopping")
            break
        if epochs_without_improvement >= cfg.early_stopping_patience:
            logger.info("Early stopping at epoch %d (best epoch %d)", epoch, history.best_epoch)
            break

    # Restore the weights from the best validation epoch, not the last one.
    model.load_state_dict(best_state)
    history.seconds = time.perf_counter() - started
    return history
