from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import numpy as np
from cmaes import CMA

if TYPE_CHECKING:
    from lib.callbacks import CMAESMetricsCollector

from lib.stopping import CMAESEarlyStopping
from lib.util import EvalCounter

from .base import Optimizer


@dataclass
class CMAESState:
    mean: np.ndarray
    population_evaluations: float
    counter: EvalCounter

    @property
    def num_evaluations(self):
        return self.counter.num_evaluations

    @property
    def best_solutions(self):
        return self.counter.best_solutions


class CMAES(Optimizer):
    def __init__(
        self,
        fun: EvalCounter,
        mean: np.ndarray,
        popsize: int,
        seed: int,
        stopper: CMAESEarlyStopping,
        sigma: float = 1,
    ):
        self.inner = CMA(mean=mean, sigma=sigma, seed=seed, population_size=popsize)
        self.seed = seed
        self.stopper = stopper
        self.state = CMAESState(
            mean=np.array([]),
            population_evaluations=[],  # pyright: ignore[reportArgumentType]
            counter=fun,
        )

    @property
    def raw_objective(self):
        """Unwrap the objective function from the EvalCounter."""
        return self.state.counter.fun

    @property
    def wrapped_objective(self):
        return self.state.counter

    def update_state(self, population_evaluations: list[float]):
        self.state.population_evaluations = population_evaluations
        self.state.mean = self.mean

    def step(self):
        solutions = []
        for _ in range(self.inner.population_size):
            x = self.inner.ask()
            solutions.append((x, self.wrapped_objective(x)))

        self.inner.tell(solutions)
        self.update_state([sol[1] for sol in solutions])

        return solutions

    @override
    def optimize(self, callback: "CMAESMetricsCollector"):
        while not self.stopper(self.state):
            solutions = self.step()
            callback(self.state)

    @property
    def mean(self):
        return self.inner.mean
