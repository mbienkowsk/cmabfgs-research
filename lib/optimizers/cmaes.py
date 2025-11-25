from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

import numpy as np
from cmaes import CMA

if TYPE_CHECKING:
    from lib.metrics_collector import MetricsCollector

from lib.bound_handling import RepairMethod, repair_by_reflection
from lib.stopping import CMAESEarlyStopping
from lib.util import EvalCounter

from .base import Optimizer


@dataclass
class CMAESState:
    counter: EvalCounter
    covariance_matrix: np.ndarray
    mean: np.ndarray = field(default_factory=lambda: np.array([]))
    population_evaluations: list[float] = field(default_factory=list)

    @property
    def num_evaluations(self):
        return self.counter.num_evaluations

    @property
    def best_solutions(self):
        return self.counter.best_solutions

    @property
    def best_so_far(self):
        return (
            self.counter.best_solutions[-1] if self.counter.best_solutions else np.inf
        )


class CMAES(Optimizer):
    def __init__(
        self,
        fun: EvalCounter,
        mean: np.ndarray,
        popsize: int,
        seed: int,
        stopper: CMAESEarlyStopping,
        callback: "MetricsCollector",
        bounds: tuple[int, int],
        sigma: float = 1,
        repair_method: RepairMethod = RepairMethod.REFLECT,
        identifier: str = "",
    ):
        self.inner = CMA(mean=mean, sigma=sigma, seed=seed, population_size=popsize)
        self.seed = seed
        self.stopper = stopper
        self.state = CMAESState(
            covariance_matrix=self.inner._C,
            counter=fun,
        )
        self.callback = callback
        self.bounds = bounds
        self.identifier = identifier

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
        self.state.covariance_matrix = self.inner._C

    @property
    def C(self):
        return self.inner._C

    @property
    def evals_remaining(self):
        if not self.stopper.max_evals:
            return int("inf")
        return self.stopper.max_evals - self.state.num_evaluations

    def _step(self):
        solutions = []
        for _ in range(self.inner.population_size):
            x = self.inner.ask()
            repaired = repair_by_reflection(x, self.bounds)
            solutions.append((repaired, self.wrapped_objective(repaired)))

        self.inner.tell(solutions)
        self.update_state([sol[1] for sol in solutions])

        return deepcopy(solutions)

    def step(self):
        solutions = self._step()
        self.callback(self.state, self.identifier)
        return solutions

    @override
    def optimize(self):
        while not self.stopper(self.state, self.identifier):
            _ = self.step()

    @property
    def mean(self):
        return self.inner.mean
