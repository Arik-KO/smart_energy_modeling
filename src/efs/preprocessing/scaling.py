"""Min-max scaling fitted on the training period only."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class MinMaxScaler1D:
    """Scale one series to [0, 1] using the training minimum and maximum.

    Validation and test values may fall outside [0, 1] (e.g. a colder
    December). That is expected and correct: re-fitting on them would leak
    information about the future into the model's inputs.
    """

    minimum: float = 0.0
    maximum: float = 1.0

    @classmethod
    def fit(cls, values: np.ndarray) -> "MinMaxScaler1D":
        values = np.asarray(values, dtype=float)
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if maximum - minimum < 1e-12:
            raise ValueError("Cannot scale a constant series")
        return cls(minimum=minimum, maximum=maximum)

    @property
    def span(self) -> float:
        return self.maximum - self.minimum

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.minimum) / self.span

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.span + self.minimum

    def save(self, path: Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MinMaxScaler1D":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))
