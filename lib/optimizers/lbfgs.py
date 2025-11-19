from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from scipy.optimize import Bounds, OptimizeResult, minimize

from lib.stopping import BFGSEarlyStopping, StopOptimization

if TYPE_CHECKING:
    from lib.metrics_collector import MetricsCollector

from lib.optimizers.base import Optimizer
from lib.util import EvalCounter

LBFGS_GTOL = 1e-12


@dataclass
class L_BFGS_BState:
    counter: EvalCounter
    current_result: OptimizeResult | None = None

    @property
    def num_evaluations(self):
        return self.counter.num_evaluations

    @property
    def best_solutions(self):
        return self.counter.best_solutions


class L_BFGS_B(Optimizer):
    state: L_BFGS_BState

    def __init__(
        self,
        x0: np.ndarray,
        fun: EvalCounter,
        callback: "MetricsCollector",
        stopper: BFGSEarlyStopping,
        bounds: tuple[int, int],
        identifier: str = "",
    ):
        self.x0 = x0
        self.inner = None
        self.state = L_BFGS_BState(fun)
        self.stopper = stopper
        self.callback = callback
        self.bounds = bounds
        self.identifier = identifier

    def optimize(self):
        def callback_wrapper(intermediate_result: OptimizeResult):
            self.state.current_result = intermediate_result
            # bounds handled by scipy
            self.stopper(self.state)
            return self.callback(self.state, self.identifier)

        try:
            result = minimize(
                self.state.counter,
                self.x0,
                method="L-BFGS-B",
                callback=callback_wrapper,
                bounds=Bounds(self.bounds[0], self.bounds[1], keep_feasible=False),
                options={
                    "gtol": LBFGS_GTOL,
                },
            )

            if not result.success:
                logger.warning(
                    f"L-BFGS-B {self.identifier} did not converge: {result.message}"
                )
            else:
                logger.debug(
                    f"L-BFGS-B {self.identifier} converged successfully at {self.state.best_solutions[-1]}, message: {result.message}"
                )

        except StopOptimization as e:
            logger.info(f"L-BFGS-B {self.identifier} stopped early: {e}")
