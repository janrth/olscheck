"""Core replenishment simulation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .policies import HistoricalErrorSafetyStockPolicy


@dataclass
class SimulationConfig:
    """Configuration for the replenishment simulation."""

    initial_stock: float = 0.0
    lead_time_days: int = 1
    time_step_days: int = 1
    holding_cost_per_unit_per_day: float = 0.0
    stockout_cost_per_unit_per_day: float = 0.0
    order_up_to_multiplier: float = 1.0
    forecast_window: int = 5


class ReplenishmentSimulation:
    """Runs an inventory replenishment simulation over a demand series."""

    def __init__(
        self,
        config: SimulationConfig,
        policy: HistoricalErrorSafetyStockPolicy | None = None,
    ) -> None:
        self.config = config
        self.policy = policy or HistoricalErrorSafetyStockPolicy()
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.time_step_days <= 0:
            raise ValueError("time_step_days must be positive")
        if self.config.lead_time_days <= 0:
            raise ValueError("lead_time_days must be positive")
        if self.config.lead_time_days % self.config.time_step_days != 0:
            raise ValueError("lead_time_days must be divisible by time_step_days")
        if self.config.forecast_window <= 0:
            raise ValueError("forecast_window must be positive")

    def run(self, demand: Iterable[float]) -> list[dict[str, float]]:
        """Run the simulation.

        Args:
            demand: Iterable of demand per time step.

        Returns:
            List of dicts with simulation state per step.
        """

        demand_series = list(demand)
        lead_time_steps = self.config.lead_time_days // self.config.time_step_days
        on_hand = float(self.config.initial_stock)
        backlog = 0.0
        pipeline: list[tuple[int, float]] = []
        error_history: list[float] = []
        results: list[dict[str, float]] = []

        for step, actual_demand in enumerate(demand_series):
            arrivals = [qty for arrival_step, qty in pipeline if arrival_step == step]
            if arrivals:
                on_hand += sum(arrivals)
            pipeline = [item for item in pipeline if item[0] > step]

            forecast = self._forecast(demand_series[:step])
            if step > 0:
                error_history.append(demand_series[step - 1] - self._forecast(demand_series[: step - 1]))

            safety_stock = self.policy.safety_stock(error_history)
            forecast_lead_time = forecast * lead_time_steps
            order_up_to = (forecast_lead_time + safety_stock) * self.config.order_up_to_multiplier
            inventory_position = on_hand + sum(qty for _, qty in pipeline) - backlog

            order_qty = max(order_up_to - inventory_position, 0.0)
            if order_qty > 0:
                arrival_step = step + lead_time_steps
                pipeline.append((arrival_step, order_qty))

            demand_fulfilled = min(on_hand, actual_demand)
            on_hand -= demand_fulfilled
            backlog = max(actual_demand - demand_fulfilled, 0.0)

            holding_cost = on_hand * self.config.holding_cost_per_unit_per_day * self.config.time_step_days
            stockout_cost = backlog * self.config.stockout_cost_per_unit_per_day * self.config.time_step_days

            results.append(
                {
                    "step": float(step),
                    "demand": float(actual_demand),
                    "forecast": float(forecast),
                    "safety_stock": float(safety_stock),
                    "order_qty": float(order_qty),
                    "on_hand": float(on_hand),
                    "backlog": float(backlog),
                    "holding_cost": float(holding_cost),
                    "stockout_cost": float(stockout_cost),
                    "total_cost": float(holding_cost + stockout_cost),
                }
            )

        return results

    def _forecast(self, history: list[float]) -> float:
        if not history:
            return 0.0
        window = history[-self.config.forecast_window :]
        return sum(window) / len(window)
