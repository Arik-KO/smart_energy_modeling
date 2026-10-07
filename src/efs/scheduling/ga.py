"""Genetic algorithm (GA) for the daily battery schedule.

Chromosome: one gene per day = the requested battery energy for that day
(kWh; > 0 charge, < 0 discharge), limited to [-capacity, +capacity].
Fitness:    the net cost from ``simulate`` (lower is better), which already
applies every physical limit and the end-of-horizon refill cost.

Operators (as described in the paper): tournament selection, two-point
crossover, Gaussian mutation, and elitism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from efs.scheduling.battery import BatterySpec, simulate
from efs.scheduling.tariff import Tariff


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 50
    generations: int = 100
    tournament_size: int = 3
    crossover_prob: float = 0.8
    mutation_rate: float = 0.1       # probability that a gene is mutated
    mutation_std_kwh: float = 5.0
    elitism: int = 1

    @classmethod
    def from_config(cls, ga_cfg: dict) -> "GAConfig":
        return cls(
            population_size=int(ga_cfg["population_size"]),
            generations=int(ga_cfg["generations"]),
            tournament_size=int(ga_cfg["tournament_size"]),
            crossover_prob=float(ga_cfg["crossover_prob"]),
            mutation_rate=float(ga_cfg["mutation_rate"]),
            mutation_std_kwh=float(ga_cfg["mutation_std_kwh"]),
            elitism=int(ga_cfg["elitism"]),
        )


def _cost(genes: np.ndarray, demand: np.ndarray, pv: np.ndarray, battery: BatterySpec,
          tariff: Tariff, initial_soc: float | None) -> float:
    return simulate(demand, pv, genes, battery, tariff, initial_soc).net_cost


def _tournament(costs: np.ndarray, size: int, rng: np.random.Generator) -> int:
    """Pick ``size`` random individuals; return the index of the cheapest."""
    contenders = rng.choice(len(costs), size=size, replace=False)
    best = contenders[0]
    for index in contenders[1:]:
        if costs[index] < costs[best]:
            best = index
    return int(best)


def _two_point_crossover(parent_a: np.ndarray, parent_b: np.ndarray,
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Swap the genes between two random cut points."""
    n = len(parent_a)
    child_a = parent_a.copy()
    child_b = parent_b.copy()
    if n < 2:
        return child_a, child_b
    first, second = sorted(rng.choice(n + 1, size=2, replace=False))
    child_a[first:second] = parent_b[first:second]
    child_b[first:second] = parent_a[first:second]
    return child_a, child_b


def optimize_schedule(demand_kwh: np.ndarray, pv_kwh: np.ndarray, battery: BatterySpec,
                      tariff: Tariff, cfg: GAConfig, rng: np.random.Generator,
                      initial_soc_kwh: float | None = None,
                      seed_individuals: list[np.ndarray] | None = None) -> tuple[np.ndarray, float]:
    """Return the best request schedule found and its net cost on the given inputs.

    ``seed_individuals`` (e.g. "do nothing" and the self-consumption rule) are
    placed in the first generation, so the GA can never end worse than them.
    """
    demand_kwh = np.asarray(demand_kwh, dtype=float)
    pv_kwh = np.asarray(pv_kwh, dtype=float)
    n_days = len(demand_kwh)
    bound = battery.capacity_kwh

    population = rng.uniform(-bound, bound, size=(cfg.population_size, n_days))
    for position, individual in enumerate(seed_individuals or []):
        if position < cfg.population_size:
            population[position] = np.clip(individual, -bound, bound)

    costs = np.empty(cfg.population_size)
    for index in range(cfg.population_size):
        costs[index] = _cost(population[index], demand_kwh, pv_kwh, battery, tariff, initial_soc_kwh)

    for _ in range(cfg.generations):
        order = np.argsort(costs)
        next_population = [population[index].copy() for index in order[:cfg.elitism]]

        while len(next_population) < cfg.population_size:
            parent_a = population[_tournament(costs, cfg.tournament_size, rng)]
            parent_b = population[_tournament(costs, cfg.tournament_size, rng)]
            if rng.random() < cfg.crossover_prob:
                child_a, child_b = _two_point_crossover(parent_a, parent_b, rng)
            else:
                child_a, child_b = parent_a.copy(), parent_b.copy()
            for child in (child_a, child_b):
                mutate = rng.random(n_days) < cfg.mutation_rate
                child[mutate] += rng.normal(0.0, cfg.mutation_std_kwh, size=int(mutate.sum()))
                np.clip(child, -bound, bound, out=child)
                if len(next_population) < cfg.population_size:
                    next_population.append(child)

        population = np.array(next_population)
        for index in range(cfg.population_size):
            costs[index] = _cost(population[index], demand_kwh, pv_kwh, battery, tariff,
                                 initial_soc_kwh)

    best = int(np.argmin(costs))
    return population[best].copy(), float(costs[best])
