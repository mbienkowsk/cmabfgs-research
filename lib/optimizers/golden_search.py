from dataclasses import dataclass
from typing import override

import numpy as np
from loguru import logger
from scipy.optimize import golden

from lib.bound_handling import check_bounds
from lib.metrics_collector import MetricsCollector
from lib.optimizers.base import Optimizer
from lib.util import EvalCounter, one_dimensional


@dataclass
class GoldenSearchState:
    counter: EvalCounter


class GoldenSearch(Optimizer):
    def __init__(
        self,
        x: np.ndarray,
        direction: np.ndarray,
        fun: EvalCounter,
        callback: MetricsCollector,
        bounds: tuple[int, int] = (-100, 100),
        identifier="",
    ):
        self.x0 = x
        self.direction = direction / np.linalg.norm(direction)
        self.fun = fun
        self.callback = callback
        self.bounds = bounds
        self.identifier = identifier
        self.state = GoldenSearchState(fun)

    @override
    def optimize(self):
        solution, fval, funcalls = golden(
            one_dimensional(self.fun, self.x0, self.direction), full_output=True
        )
        if check_bounds(solution, self.bounds, False):
            self.callback(self.state, self.identifier)
        else:
            logger.warning("Golden search produced out-of-bounds solution.")
