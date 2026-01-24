"""Policy implementations for replenishment decisions."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev


@dataclass
class HistoricalErrorSafetyStockPolicy:
    """Safety stock based on historical forecast errors.

    safety_stock = safety_factor * stddev(error_history)
    """

    safety_factor: float = 1.0
    minimum_samples: int = 3

    def safety_stock(self, error_history: list[float]) -> float:
        if len(error_history) < self.minimum_samples:
            return 0.0
        return self.safety_factor * pstdev(error_history)
