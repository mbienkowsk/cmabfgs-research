from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from lib.optimizers.cmaes import CMAESState


class StopReason(Enum):
    NOT_GIVEN = auto()
    MAXEVALS = auto()
    TOLFUN = auto()
    TOLHIST = auto()


class StopOptimization(Exception):
    """Used to signal an optimizer to stop looking for the solution
    and return the control flow to the caller"""

    def __init__(self, reason=StopReason.NOT_GIVEN):
        self.reason = reason
        super().__init__(f"Optimization stopped: {reason.name}")


@dataclass
class CMAESEarlyStopping:
    max_evals: int | None = field(default=None)
    tolfun: float | None = field(default=None)

    def check_max_evals(self, state: CMAESState):
        return self.max_evals is not None and state.num_evaluations >= self.max_evals

    def check_tolfun(self, state: CMAESState):
        if self.tolfun is None or not state.population_evaluations:
            return False

        best, worst = np.min(state.population_evaluations), np.max(
            state.population_evaluations
        )
        return np.abs(best - worst) < self.tolfun

    def __call__(self, state: CMAESState):
        return any(
            (
                self.check_max_evals(state),
                self.check_tolfun(state),
            )
        )
