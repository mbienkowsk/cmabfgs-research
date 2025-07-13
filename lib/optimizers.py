from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, override

import numpy as np
from cmaes import CMA
from scipy.optimize import golden, minimize

if TYPE_CHECKING:
    from lib.callbacks import ExperimentCallback

from lib.util import StopOptimization, gradient_central, one_dimensional


class Optimizer(ABC):
    """Common interface for all optimizers"""

    @abstractmethod
    def step(self, objective: Callable): ...

    @abstractmethod
    @abstractmethod
    def _optimize(self, objective: Callable, callback: "ExperimentCallback"): ...

    def optimize(self, objective: Callable, callback: "ExperimentCallback"):
        try:
            self._optimize(objective, callback)
        except StopOptimization:
            return


class CMAES(Optimizer):
    def __init__(
        self,
        mean: np.ndarray,
        popsize: int,
        seed: int,
        sigma: float = 1,
    ):
        self.inner = CMA(mean=mean, sigma=sigma, seed=seed, population_size=popsize)
        self.seed = seed

    def step(self, objective: Callable):
        solutions = []
        for _ in range(self.inner.population_size):
            x = self.inner.ask()
            solutions.append((x, objective(x)))

        self.inner.tell(solutions)

    @override
    def _optimize(self, objective: Callable, callback: "ExperimentCallback"):
        while True:
            self.step(objective)
            callback(self.inner)

    @property
    def mean(self):
        return self.inner.mean


class BFGS(Optimizer):
    def __init__(self, x0: np.ndarray, seed: int):
        self.x0 = x0
        self.inner = None

    def _optimize(self, objective: Callable, callback: "ExperimentCallback"):
        minimize(
            objective,
            self.x0,
            method="BFGS",
            callback=callback,
        )


class LBFGS(BFGS):
    def _optimize(
        self, objective: Callable, callback: "ExperimentCallback", fruppo: int
    ):
        minimize(
            objective,
            self.x0,
            method="L-BFGS-B",
            callback=callback,
        )


class CMABFGSHybrid(Optimizer):
    def __init__(
        self,
        x0: np.ndarray,
        n_cmaes_iterations: int,
        seed: int,
        popsize: int,
        sigma: int = 1,
    ):
        self.cma = CMA(mean=x0, sigma=sigma, seed=seed, population_size=popsize)
        self.seed = seed
        self.n_cmaes_iterations = n_cmaes_iterations

    def cma_step(self, objective: Callable):
        solutions = []
        for _ in range(self.cma.population_size):
            x = self.cma.ask()
            solutions.append((x, objective(x)))
            self.cma.tell(solutions)

        self.cma.tell(solutions)

    def _optimize(self, objective: Callable, callback: "ExperimentCallback"):

        for _ in range(self.n_cmaes_iterations):
            self.cma_step(objective)
            callback(self.cma)

        minimize(
            objective,
            self.cma.mean,
            method="BFGS",
            callback=callback,
        )


class CMAGoldenSearchHybrid(Optimizer):

    def __init__(
        self,
        x0: np.ndarray,
        n_cmaes_iterations: int,
        seed: int,
        popsize: int,
        sigma: int = 1,
    ):
        self.cma = CMA(mean=x0, sigma=sigma, seed=seed, population_size=popsize)
        self.seed = seed
        self.n_cmaes_iterations = n_cmaes_iterations

    def cma_step(self, objective: Callable):
        solutions = []
        for _ in range(self.cma.population_size):
            x = self.cma.ask()
            solutions.append((x, objective(x)))
            self.cma.tell(solutions)

        self.cma.tell(solutions)

    def _optimize(self, objective: Callable, callback: "ExperimentCallback"):

        for _ in range(self.n_cmaes_iterations):
            self.cma_step(objective)
            callback(self.cma)

        d = self.cma._C @ gradient_central(objective, self.cma.mean)

        solution, fval, funcalls = golden(
            one_dimensional(objective, self.cma.mean, d), full_output=True
        )
