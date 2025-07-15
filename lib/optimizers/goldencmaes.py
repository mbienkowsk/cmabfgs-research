from typing import TYPE_CHECKING

import numpy as np
from cmaes import CMA
from scipy.optimize import golden

if TYPE_CHECKING:
    from lib.callbacks import ExperimentCallback

from lib.optimizers.base import Optimizer
from lib.stopping import CMAESEarlyStopping
from lib.util import EvalCounter, gradient_central, one_dimensional


# TODO: pass bfgs callback as well
class GoldenCMAES(Optimizer):

    def __init__(
        self,
        x0: np.ndarray,
        n_cmaes_iterations: int,
        seed: int,
        popsize: int,
        fun: EvalCounter,
        stopping: CMAESEarlyStopping,
        sigma: int = 1,
    ):
        self.cma = CMA(mean=x0, sigma=sigma, seed=seed, population_size=popsize)
        self.seed = seed
        self.n_cmaes_iterations = n_cmaes_iterations
        self.fun = fun
        self.stopping = stopping

    def cma_step(self):
        solutions = []
        for _ in range(self.cma.population_size):
            x = self.cma.ask()
            solutions.append((x, self.fun(x)))
            self.cma.tell(solutions)

        self.cma.tell(solutions)

    def optimize(self, callback: "ExperimentCallback"):
        for _ in range(self.n_cmaes_iterations):
            if self.stopping(self.state):
                return

            self.cma_step()
            callback(self.cma)

        d = self.cma._C @ gradient_central(self.fun, self.cma.mean)

        solution, fval, funcalls = golden(
            one_dimensional(self.fun, self.cma.mean, d), full_output=True
        )
