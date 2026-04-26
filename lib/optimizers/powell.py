from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from scipy.optimize import Bounds, OptimizeResult, minimize

from lib.optimizers.base import Optimizer
from lib.stopping import PowellEarlyStopping, StopOptimization
from lib.util import EvalCounter

POWELL_XTOL = 1e-15
POWELL_FTOL = 1e-15
POWELL_MAXFEV = int(1e9)

if TYPE_CHECKING:
    from lib.metrics_collector import MetricsCollector


@dataclass
class PowellState:
    counter: EvalCounter
    current_x: np.ndarray | None = None
    end_result: OptimizeResult | None = None

    @property
    def num_evaluations(self):
        return self.counter.num_evaluations

    @property
    def best_solutions(self):
        return self.counter.best_solutions


class Powell(Optimizer):
    state: PowellState

    def __init__(
        self,
        x0: np.ndarray,
        fun: EvalCounter,
        callback: "MetricsCollector",
        stopper: PowellEarlyStopping,
        bounds: tuple[float, float],
        direc0: np.ndarray | None = None,
        identifier: str = "",
    ):
        self.x0 = x0
        self.state = PowellState(fun)
        self.stopper = stopper
        self.callback = callback
        self.bounds = Bounds(lb=bounds[0], ub=bounds[1])
        self.direc0 = direc0
        self.identifier = identifier

    def optimize(self):
        def callback_wrapper(x: np.ndarray):
            self.state.current_x = x
            self.stopper(self.state)
            return self.callback(self.state, self.identifier)

        try:
            self.state.counter(self.x0)
            self.callback(self.state, self.identifier)
            result = minimize(
                self.state.counter,
                self.x0,
                method="Powell",
                bounds=self.bounds,
                callback=callback_wrapper,
                options={
                    "direc": self.direc0,
                    "xtol": POWELL_XTOL,
                    "ftol": POWELL_FTOL,
                    "maxfev": POWELL_MAXFEV,
                },
            )
            self.state.end_result = result
            if not result.success:
                logger.warning(
                    f"Powell {self.identifier} did not converge: {result.message}"
                )
            else:
                logger.debug(
                    f"Powell {self.identifier} converged successfully: {result.message}"
                )
        except StopOptimization as e:
            logger.info(f"Powell {self.identifier} stopped early: {e}")
