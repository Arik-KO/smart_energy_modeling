"""Two-step (tiered) residential tariff applied to daily net grid energy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tariff:
    """BC Hydro 2019 step rate, pro-rated to a daily threshold (following [12]).

    Positive daily net energy is imported and billed in two steps;
    negative daily net energy is exported and credited at ``export_price``.
    """

    tier1_price: float
    tier2_price: float
    daily_threshold_kwh: float
    export_price: float

    @classmethod
    def from_config(cls, tariff_cfg: dict) -> "Tariff":
        return cls(
            tier1_price=float(tariff_cfg["tier1_price"]),
            tier2_price=float(tariff_cfg["tier2_price"]),
            daily_threshold_kwh=float(tariff_cfg["daily_threshold_kwh"]),
            export_price=float(tariff_cfg["export_price"]),
        )

    def split_import(self, net_kwh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split each day's import into the Tier-1 part and the Tier-2 part (kWh)."""
        imports = np.maximum(net_kwh, 0.0)
        tier1_kwh = np.minimum(imports, self.daily_threshold_kwh)
        tier2_kwh = imports - tier1_kwh
        return tier1_kwh, tier2_kwh

    def import_cost(self, net_kwh: np.ndarray) -> np.ndarray:
        """Daily cost of imported energy ($)."""
        tier1_kwh, tier2_kwh = self.split_import(np.asarray(net_kwh, dtype=float))
        return tier1_kwh * self.tier1_price + tier2_kwh * self.tier2_price

    def export_revenue(self, net_kwh: np.ndarray) -> np.ndarray:
        """Daily credit for exported energy ($)."""
        exports = np.maximum(-np.asarray(net_kwh, dtype=float), 0.0)
        return exports * self.export_price
