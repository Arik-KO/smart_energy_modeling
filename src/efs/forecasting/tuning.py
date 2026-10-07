"""Hyperparameter tuning with Optuna (TPE sampler + median pruner).

Every architecture/target pair gets the same search budget and the same
objective: the November month-ahead error from ``train_and_validate``.
Studies are stored in SQLite, so an interrupted search (power cut, closed
laptop) resumes where it stopped instead of starting over.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import optuna
import pandas as pd
import torch
import yaml

from efs.config import resolve_path
from efs.forecasting.models import base_architecture
from efs.forecasting.pipeline import train_and_validate
from efs.forecasting.train import TrainingConfig

logger = logging.getLogger(__name__)

# Keys in the search space that belong to training, not to the network.
TRAINING_KEYS = ("learning_rate", "weight_decay")
WINDOW_KEY = "window_hours"
FINISHED_STATES = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                   optuna.trial.TrialState.FAIL)


def suggest(trial: optuna.Trial, name: str, spec: dict):
    """Ask Optuna for one value, following a spec from the config."""
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if kind == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]),
                                   log=bool(spec.get("log", False)))
    raise ValueError(f"Unknown search-space type '{kind}' for '{name}'")


def sample_settings(trial: optuna.Trial, config: dict, architecture: str) -> tuple[dict, dict, int]:
    """Draw (model params, training overrides, window length) for one trial."""
    space = config["tuning"]["search_space"]
    base = base_architecture(architecture)

    # Start from the defaults (e.g. pooling for the Transformer), then override.
    model_params = dict(config["models"][base])
    training_overrides = {}
    window_hours = None

    specs = dict(space["common"])
    specs.update(space[base])
    for name, spec in specs.items():
        value = suggest(trial, name, spec)
        if name == WINDOW_KEY:
            window_hours = int(value)
        elif name in TRAINING_KEYS:
            training_overrides[name] = float(value)
        else:
            model_params[name] = value

    if window_hours is None:
        window_hours = int(config["forecasting"]["window_hours"])
    return model_params, training_overrides, window_hours


def make_objective(dataset: pd.DataFrame, config: dict, target: str, architecture: str,
                   device: torch.device):
    tuning_cfg = config["tuning"]
    strategy = tuning_cfg["strategy"]
    seed = int(tuning_cfg["training_seed"])
    base_training = TrainingConfig.from_config(config["training"])

    def objective(trial: optuna.Trial) -> float:
        model_params, training_overrides, window_hours = sample_settings(trial, config, architecture)
        training_cfg = dataclasses.replace(
            base_training,
            max_epochs=int(tuning_cfg["max_epochs"]),
            log_every=10_000,             # keep the log readable: one line per trial
            **training_overrides,
        )

        def report_epoch(epoch: int, val_loss: float) -> None:
            # The pruner compares this trial's validation loss with other
            # trials at the same epoch and stops clearly worse ones early.
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch}")

        try:
            result = train_and_validate(
                dataset, config, target, architecture, strategy, seed, device,
                params=model_params, window_hours=window_hours,
                training_cfg=training_cfg, epoch_callback=report_epoch,
            )
        except torch.cuda.OutOfMemoryError:
            # A setting too large for the GPU is recorded as pruned, not fatal.
            torch.cuda.empty_cache()
            logger.warning("Trial %d ran out of GPU memory; recorded as pruned", trial.number)
            raise optuna.TrialPruned("out of GPU memory")

        # Keep the details with the trial for the paper's tuning table.
        trial.set_user_attr("best_epoch", result.history.best_epoch)
        trial.set_user_attr("train_seconds", round(result.history.seconds, 1))
        for key in ("hourly_mape", "daily_mape", "hourly_rmse", "skill_vs_naive"):
            trial.set_user_attr(key, float(result.val_report[key]))
        return result.objective

    return objective


def open_study(config: dict, target: str, architecture: str) -> optuna.Study:
    """Create or reopen the persistent study for one target/architecture pair."""
    tuning_cfg = config["tuning"]
    storage_dir = resolve_path(config, tuning_cfg["storage_dir"])
    storage_dir.mkdir(parents=True, exist_ok=True)
    name = f"{target}_{architecture}_{tuning_cfg['strategy']}"
    storage = f"sqlite:///{(storage_dir / f'{name}.db').as_posix()}"

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(tuning_cfg["pruner_startup_trials"]),
        n_warmup_steps=int(tuning_cfg["pruner_warmup_epochs"]),
    )
    study = optuna.create_study(study_name=name, storage=storage, direction="minimize",
                                pruner=pruner, load_if_exists=True)

    # Seed the sampler with (base seed + trials already stored). A fresh study
    # gets the base seed; a resumed one continues with a new seed. Re-using the
    # base seed on resume would replay the first random settings and waste
    # trials on exact duplicates.
    seed = int(tuning_cfg["sampler_seed"]) + len(study.trials)
    study.sampler = optuna.samplers.TPESampler(seed=seed)
    return study


def finished_trials(study: optuna.Study) -> int:
    count = 0
    for trial in study.trials:
        if trial.state in FINISHED_STATES:
            count += 1
    return count


def run_study(dataset: pd.DataFrame, config: dict, target: str, architecture: str,
              device: torch.device) -> optuna.Study:
    """Run (or finish) the configured number of trials for one pair."""
    study = open_study(config, target, architecture)
    total = int(config["tuning"]["n_trials"])
    remaining = total - finished_trials(study)
    if remaining <= 0:
        logger.info("%s: all %d trials already done", study.study_name, total)
        return study

    logger.info("%s: running %d of %d trials", study.study_name, remaining, total)

    def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # Pruned trials carry their last validation loss as "value", not the
        # objective; show a dash so the log is not misleading.
        is_complete = trial.state == optuna.trial.TrialState.COMPLETE
        value = f"{trial.value:.3f}" if is_complete else "-"
        best = study.best_value if any(t.state == optuna.trial.TrialState.COMPLETE
                                       for t in study.trials) else float("nan")
        logger.info("%s | trial %2d | %-8s | objective %s | best so far %.3f | %s",
                    study.study_name, trial.number, trial.state.name, value, best, trial.params)

    study.optimize(make_objective(dataset, config, target, architecture, device),
                   n_trials=remaining, callbacks=[log_trial], gc_after_trial=True)
    return study


def best_settings(study: optuna.Study, config: dict, target: str, architecture: str) -> dict:
    """The winning trial's settings, in the form the final training will read."""
    best = study.best_trial
    base = base_architecture(architecture)
    model_params = dict(config["models"][base])
    training = {}
    window_hours = int(config["forecasting"]["window_hours"])
    for name, value in best.params.items():
        if name == WINDOW_KEY:
            window_hours = int(value)
        elif name in TRAINING_KEYS:
            training[name] = float(value)
        else:
            model_params[name] = value

    return {
        "target": target,
        "architecture": architecture,
        "strategy": config["tuning"]["strategy"],
        "window_hours": window_hours,
        "model_params": model_params,
        "training": training,
        "objective_metric": config["evaluation"]["objective_metric"][target],
        "objective_value": float(best.value),
        "best_trial": int(best.number),
        "n_trials": len(study.trials),
        "validation": dict(best.user_attrs),
    }


def save_best_settings(settings: dict, config: dict) -> Path:
    output_dir = resolve_path(config, config["tuning"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{settings['target']}_{settings['architecture']}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(settings, handle, sort_keys=False)
    return path
