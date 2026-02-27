from collections.abc import Iterable
from typing import TYPE_CHECKING

from lib.enums import HessianNormalization
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
        callbacks: Iterable["MetricsCollector"],
        cmaes_stopper: CMAESEarlyStopping,
        maxevals: int,
        bounds: tuple[float, float] = (-100, 100),
        sigma: int = 1,
        hess_scaling: HessianNormalization = HessianNormalization.UNIT,
        precondition_using_C: bool = True,
    ):
        self.nums_cmaes_iterations = nums_cmaes_iterations
        self.cmaes = CMAES(
            fun,
            x0,
            popsize,
            seed,
            cmaes_stopper,
            list(callbacks),
            bounds,
            sigma,
            identifier="vanilla_cmaes",
        )
        self.seed = seed
        self.fun = fun
        self.callbacks = callbacks
        self.maxevals = maxevals
        self.bounds = bounds
        self.precondition = precondition_using_C
        self.x0 = x0

    def optimize(self):
        shifted = [0] + self.nums_cmaes_iterations[:-1]
        differences = [x - y for x, y in zip(self.nums_cmaes_iterations, shifted)]
        for idx, switch_after in enumerate(differences):
            for _ in range(switch_after):
                self.cmaes.step()

            hess_inv0 = (
                (self.cmaes.C + self.cmaes.C.T) / 2
                if self.precondition
                else np.eye(self.x0.shape[0], self.x0.shape[0])
            )
            identifier = str(self.nums_cmaes_iterations[idx])
            fun = self.fun.copy_with_identifier(f"bfgs_{identifier}")
            # bfgs gets its own eval counter
            for callback in self.callbacks:
                callback(self.cmaes.state, identifier)

            bfgs = BFGS(
                self.cmaes.mean,
                fun,
                self.callbacks,
                BFGSEarlyStopping(self.maxevals),
                self.bounds,
                identifier=identifier,
                hess_inv0=hess_inv0,
            )

            for callback in self.callbacks:
                callback(bfgs.state, identifier)

            bfgs.optimize()
            self.cmaes.state.counter.num_evaluations = bfgs.state.num_evaluations

        self.cmaes.optimize()
