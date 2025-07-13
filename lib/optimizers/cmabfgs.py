from typing import TYPE_CHECKING, Callable

import numpy as np
from cmaes import CMA
from scipy.optimize import minimize

from lib.optimizers.base import Optimizer

if TYPE_CHECKING:
    from lib.callbacks import ExperimentCallback


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

        return solutions.copy()

    def optimize(self, objective: Callable, callback: "ExperimentCallback"):

        for _ in range(self.n_cmaes_iterations):
            self.cma_step(objective)
            callback(self.cma)

        minimize(
            objective,
            self.cma.mean,
            method="BFGS",
            callback=callback,
        )
