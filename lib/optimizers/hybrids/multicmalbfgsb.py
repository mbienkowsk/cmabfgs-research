from typing import TYPE_CHECKING

from loguru import logger

from lib.optimizers.base import Optimizer
from lib.optimizers.cmaes import CMAES
from lib.optimizers.lbfgs import L_BFGS_B
from lib.stopping import BFGSEarlyStopping, CMAESEarlyStopping

if TYPE_CHECKING:
    from lib.metrics_collector import MetricsCollector
import numpy as np

from lib.util import EvalCounter


class MultiCMALBFGSB(Optimizer):
    def __init__(
        self,
        x0: np.ndarray,
        nums_cmaes_iterations: list[int],
        seed: int,
        fun: EvalCounter,
        popsize: int,
        callback: "MetricsCollector",
        cmaes_stopper: CMAESEarlyStopping,
        maxevals: int,
        bounds: tuple[float, float] = (-100, 100),
        sigma: int = 1,
        restart_cmaes: bool = False,
    ):
        self.nums_cmaes_iterations = nums_cmaes_iterations
        self.callback = callback
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
        self.popsize = popsize
        self.seed = seed
        self.fun = fun
        self.callback = callback
        self.maxevals = maxevals
        self.bounds = bounds
        self.restart_cmaes = restart_cmaes
        self.sigma = sigma

    def optimize(self):
        shifted = [0] + self.nums_cmaes_iterations[:-1]
        differences = [x - y for x, y in zip(self.nums_cmaes_iterations, shifted)]
        for idx, switch_after in enumerate(differences):
            for _ in range(switch_after):
                self.cmaes.step()

            identifier = str(self.nums_cmaes_iterations[idx])
            fun = self.fun.copy_with_identifier(
                f"lbfgsb_{identifier}"
            )  # lbfgsb gets its own eval counter

            # cheat a little to log the starting point, this gets saved to lbfgsb's column
            self.callback(self.cmaes.state, identifier)

            lbfgsb = L_BFGS_B(
                self.cmaes.mean,
                fun,
                self.callback,
                BFGSEarlyStopping(self.maxevals),
                self.bounds,
                identifier=identifier,
            )
            lbfgsb.optimize()
            self.callback(lbfgsb.state, identifier)
            self.cmaes.state.counter.num_evaluations = lbfgsb.state.num_evaluations

            if self.restart_cmaes:
                logger.debug(
                    f"L-BFGS-B {identifier}: Restarting CMA-ES, y={lbfgsb.y}, evals remaining={self.cmaes.evals_remaining}"
                )
                es = CMAES(
                    fun,
                    lbfgsb.x,
                    self.popsize,
                    self.seed,
                    CMAESEarlyStopping(self.cmaes.evals_remaining),
                    self.callback,
                    self.bounds,
                    self.sigma,
                    identifier=identifier,
                )
                es.optimize()
                logger.debug(
                    f"L-BFGS-B {identifier}: second iteration of CMA-ES finished at y={es.state.counter.best_so_far[1]}"
                )

        self.cmaes.optimize()
