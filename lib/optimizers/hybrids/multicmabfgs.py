from typing import TYPE_CHECKING

from lib.optimizers.bfgs import BFGS
from lib.optimizers.cmaes import CMAES
from lib.stopping import BFGSEarlyStopping, CMAESEarlyStopping

if TYPE_CHECKING:
    from lib.metrics_collector import MetricsCollector
import numpy as np

from lib.optimizers.base import Optimizer
from lib.util import EvalCounter


class MultiCMABFGS(Optimizer):
    def __init__(
        self,
        x0: np.ndarray,
        nums_cmaes_iterations: list[int],
        seed: int,
        fun: EvalCounter,
        popsize: int,
        callback: "MetricsCollector",
        cmaes_stopper: CMAESEarlyStopping,
        bounds: tuple[int, int] = (-100, 100),
        sigma: int = 1,
    ):
        self.nums_cmaes_iterations = nums_cmaes_iterations
        self.cmaes = CMAES(
            fun,
            x0,
            popsize,
            seed,
            cmaes_stopper,
            callback,
            bounds,
            sigma,
            identifier="vanilla_cmaes",
        )
        self.seed = seed
        self.fun = fun
        self.callback = callback
        self.bounds = bounds

    def optimize(self):
        shifted = [0] + self.nums_cmaes_iterations[:-1]
        differences = [x - y for x, y in zip(self.nums_cmaes_iterations, shifted)]
        for idx, switch_after in enumerate(differences):
            for _ in range(switch_after):
                self.cmaes.step()

            identifier = str(self.nums_cmaes_iterations[idx])
            bfgs = BFGS(
                self.cmaes.mean,
                self.fun.copy_with_identifier(
                    f"bfgs_{identifier}"
                ),  # bfgs gets its own eval counter
                self.callback,
                BFGSEarlyStopping(self.cmaes.evals_remaining),
                self.bounds,
                identifier=identifier,
                hess_inv0=(self.cmaes.C + self.cmaes.C.T) / 2,
            )
            self.callback(bfgs.state, identifier)
            # TODO: might wanna artifically add a point here
            bfgs.optimize()

        self.cmaes.optimize()
