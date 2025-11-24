from typing import TYPE_CHECKING

import numpy as np

from lib.optimizers.cmaes import CMAES
from lib.optimizers.golden_search import GoldenSearch

if TYPE_CHECKING:
    from lib.metrics_collector import MetricsCollector

from lib.optimizers.base import Optimizer
from lib.stopping import CMAESEarlyStopping
from lib.util import EvalCounter, gradient_central


class GoldenCMAES(Optimizer):
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
        self.seed = seed
        self.fun = fun
        self.bounds = bounds

    def optimize(self):
        shifted = [0] + self.nums_cmaes_iterations[:-1]
        differences = [x - y for x, y in zip(self.nums_cmaes_iterations, shifted)]
        for idx, switch_after in enumerate(differences):
            for _ in range(switch_after):
                self.cmaes.step()

            identifier = str(self.nums_cmaes_iterations[idx])
            direction = self.cmaes.C @ gradient_central(self.fun, self.cmaes.mean)
            golden_search = GoldenSearch(
                self.cmaes.mean,
                direction,
                self.fun.copy_with_identifier(
                    f"golden_{identifier}"
                ),  # golden gets its own eval counter
                self.callback,  # FIXME: if anything besides basic counter getting will be done here, it will fail as n
                self.bounds,
                identifier=identifier,
            )
            self.callback(golden_search.state, identifier)
            golden_search.optimize()

        self.cmaes.optimize()
