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
    popsize: int
    sigma: float = 1
    mean: np.ndarray = field(default_factory=lambda: np.array([]))
    population_evaluations: list[float] = field(default_factory=list)
    population: list[np.ndarray] = field(default_factory=list)

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
        callbacks: list["MetricsCollector"],
        bounds: tuple[float, float],
        sigma: float = 1,
        repair_method: RepairMethod = RepairMethod.REFLECT,
        identifier: str = "",
    ):
        self.inner = CMA(mean=mean, sigma=sigma, seed=seed, population_size=popsize)
        self.seed = seed
        self.stopper = stopper
        self.state = CMAESState(
            covariance_matrix=self.inner._C, counter=fun, popsize=popsize
        )

        def combined_callback(state: CMAESState, identifier):
            for cb in callbacks:
                cb(state, identifier)

        self.callback = combined_callback

        self.bounds = bounds
        self.identifier = identifier

    @property
    def wrapped_objective(self):
        return self.state.counter

    def update_state(self, solutions: list[tuple[np.ndarray, float]]):
        x, y = zip(*solutions)
        self.state.population = list(x)
        self.state.population_evaluations = list(y)
        self.state.mean = self.mean
        self.state.covariance_matrix = self.inner._C
        self.state.sigma = self.inner._sigma

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

        self.update_state(deepcopy(solutions))
        self.inner.tell(solutions)

    def step(self):
        self._step()
        self.callback(self.state, self.identifier)

    @override
    def optimize(self):
        while not self.stopper(self.state, self.identifier):
            _ = self.step()

    @property
    def mean(self):
        return self.inner.mean
