from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from scipy.optimize import OptimizeResult, minimize

from lib.bound_handling import OutOfBoundsError, check_bounds
from lib.stopping import BFBGSEarlyStopping, StopOptimization

if TYPE_CHECKING:
    from lib.callbacks import MetricsCollector

from lib.optimizers.base import Optimizer
from lib.util import EvalCounter


@dataclass
class BFGSState:
    counter: EvalCounter
    current_result: OptimizeResult | None = field(default=None)

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
        seed: int,
        fun: EvalCounter,
        callback: "MetricsCollector",
        stopper: BFBGSEarlyStopping,
        bounds: tuple[int, int],
    ):
        self.x0 = x0
        self.inner = None
        self.seed = seed
        self.state = BFGSState(fun)
        self.stopper = stopper
        self.callback = callback
        self.bounds = bounds

    def optimize(self):
        self.state.counter(
            self.x0
        )  # ensures there is a single evaluation even if bfgs quits instantly
        self.callback(self.state)

        def callback_wrapper(intermediate_result: OptimizeResult):
            self.state.current_result = intermediate_result
            self.stopper(self.state)  # raises an exception
            check_bounds(
                self.state.current_result.x, self.bounds
            )  # raises an exception
            return self.callback(self.state)

        try:
            result = minimize(
                self.state.counter,
                self.x0,
                method="BFGS",
                callback=callback_wrapper,
                options={
                    "gtol": 1e-30,
                },
            )
        except StopOptimization as e:
            logger.info(f"BFGS stopped early: {e}")
        except OutOfBoundsError as e:
            logger.info(f"BFGS stopped due to out-of-bounds: {e}")

        if not result.success:
            logger.warning(f"BFGS did not converge: {result.message}")
        else:
            logger.info(f"BFGS converged successfully, message: {result.message}")
