from collections.abc import Iterable
from typing import TYPE_CHECKING, Callable

from lib.enums import HessianNormalization
from lib.optimizers.bfgs import BFGS, BFGSState
from lib.optimizers.cmaes import CMAES, CMAESState
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
        local_search_oracle: Callable[..., bool],
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
        self.local_search_oracle = local_search_oracle
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

        def combined_callback(state: CMAESState | BFGSState, identifier):
            for cb in callbacks:
                cb(state, identifier)

        self.callback = combined_callback
        self.maxevals = maxevals
        self.bounds = bounds
        self.precondition = precondition_using_C
        self.x0 = x0
        self.hess_scaling = hess_scaling

    def optimize(self):
        cmaes_iters_done = 0
        while not self.cmaes.should_stop:
            self.cmaes.step()
            cmaes_iters_done += 1

            if self.local_search_oracle(self.cmaes.state, cmaes_iters_done):
                hess_inv0 = (
                    self.cmaes.C
                    if self.precondition
                    else np.eye(self.x0.shape[0], self.x0.shape[0])
                )
                hess_inv0 = self.hess_scaling.normalize_and_make_symmetrical(hess_inv0)
                identifier = str(cmaes_iters_done)
                fun = self.fun.copy_with_identifier(f"bfgs_{identifier}")
                self.callback(self.cmaes.state, identifier)

                bfgs = BFGS(
                    self.cmaes.mean,
                    fun,
                    self.callback,  # pyright: ignore[reportArgumentType]
                    BFGSEarlyStopping(self.maxevals),
                    self.bounds,
                    identifier=identifier,
                    hess_inv0=hess_inv0,
                )

                bfgs.optimize()
                self.cmaes.state.counter.num_evaluations = bfgs.state.num_evaluations
                self.callback(bfgs.state, identifier)
