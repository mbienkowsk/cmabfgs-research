from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from cmaes import CMA
from scipy.optimize import OptimizeResult, minimize

from lib.optimizers.base import Optimizer
from lib.optimizers.bfgs import BFGSState
from lib.optimizers.cmaes import CMAESState
from lib.util import EvalCounter

if TYPE_CHECKING:
    from lib.callbacks import MetricsCollector


@dataclass
class CMABFGSState:
    mode: Literal["CMAES", "BFGS"]
    cmaes_state: CMAESState
    bfgs_state: BFGSState
    counter: EvalCounter


class CMABFGS(Optimizer):
    def __init__(
        self,
        x0: np.ndarray,
        n_cmaes_iterations: int,
        seed: int,
        fun: EvalCounter,
        popsize: int,
        callback: "MetricsCollector",
        sigma: int = 1,
    ):
        self.cma = CMA(mean=x0, sigma=sigma, seed=seed, population_size=popsize)
        self.seed = seed
        self.n_cmaes_iterations = n_cmaes_iterations
        self.callback = callback
        cmaes_state = CMAESState(fun, self.cma._C)
        bfgs_state = BFGSState(fun)
        self.state = CMABFGSState("CMAES", cmaes_state, bfgs_state, fun)

        raise NotImplementedError(
            "Add early BFGS stopping like in MultiCMABFGS before running this"
        )

    def cma_step(self):
        solutions = []
        for _ in range(self.cma.population_size):
            x = self.cma.ask()
            solutions.append((x, self.state.counter(x)))

        self.cma.tell(solutions)

        return solutions.copy()

    def optimize(self):
        for _ in range(self.n_cmaes_iterations):
            self.cma_step()
            self.callback(self.state)

        self.state.mode = "BFGS"

        def callback_wrapper(intermediate_result: OptimizeResult):
            self.state.bfgs_state.current_result = intermediate_result
            return self.callback(self.state)

        minimize(
            self.state.counter,
            self.cma.mean,
            method="BFGS",
            callback=callback_wrapper,
            options={
                "gtol": 1e-30,
                "hess_inv0": (self.cma._C + self.cma._C.T) / 2,
                # TODO: why is this still happening? investigate
            },
        )
