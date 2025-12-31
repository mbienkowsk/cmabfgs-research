from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from scipy.optimize import OptimizeResult, minimize

from lib.bound_handling import OutOfBoundsError, check_bounds
from lib.stopping import BFGSEarlyStopping, StopOptimization

if TYPE_CHECKING:
    from lib.metrics_collector import MetricsCollector

from lib.optimizers.base import Optimizer
from lib.util import EvalCounter

BFGS_GTOL = 1e-8


@dataclass
class BFGSState:
    counter: EvalCounter
    current_result: OptimizeResult | None = None

    @property
    def num_evaluations(self):
        return self.counter.num_evaluations

    @property
    def best_solutions(self):
        return self.counter.best_solutions


class BFGS(Optimizer):
    state: BFGSState

    def __init__(
        self,
        x0: np.ndarray,
        fun: EvalCounter,
        callback: "MetricsCollector",
        stopper: BFGSEarlyStopping,
        bounds: tuple[float, float],
        identifier: str = "",
        hess_inv0: np.ndarray | None = None,
    ):
        self.x0 = x0
        self.inner = None
        self.state = BFGSState(fun)
        self.stopper = stopper
        self.callback = callback
        self.bounds = bounds
        self.identifier = identifier
        self.hess_inv0 = hess_inv0

    def optimize(self):
        def callback_wrapper(intermediate_result: OptimizeResult):
            self.state.current_result = intermediate_result
            check_bounds(
                self.state.current_result.x, self.bounds
            )  # raises an exception
            self.stopper(self.state)  # raises an exception
            return self.callback(self.state, self.identifier)

        try:
            result = minimize(
                self.state.counter,
                self.x0,
                method="BFGS",
                callback=callback_wrapper,
                options={
                    "gtol": BFGS_GTOL,
                    "hess_inv0": self.hess_inv0,
                },
            )

            self.state.counter(
                np.array(self.x0)
            )  # ensures there is a single evaluation even if bfgs quits instantly
            self.callback(self.state, self.identifier)
            if not result.success:
                logger.warning(
                    f"BFGS {self.identifier} did not converge: {result.message}"
                )
            else:
                logger.debug(
                    f"BFGS {self.identifier} converged successfully, message: {result.message}"
                )
        except StopOptimization as e:
            logger.info(f"BFGS {self.identifier} stopped early: {e}")
        except OutOfBoundsError as e:
            logger.info(f"BFGS {self.identifier} stopped due to out-of-bounds: {e}")
            if len(self.callback.data) == 0:
                # bfgs was stopped immediately, crap starting point and instant oob
                # add a single point to prevent empty dataframe
                self.state.counter(self.x0)
                self.callback(self.state, self.identifier)
