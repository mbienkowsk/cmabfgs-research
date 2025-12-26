from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from lib.optimizers.bfgs import BFGSState
    from lib.optimizers.cmaes import CMAESState


class StopReason(Enum):
    NOT_GIVEN = auto()
    MAXEVALS = auto()
    TOLFUN = auto()


class HasNumEvals(Protocol):
    @property
    def num_evaluations(self) -> int: ...


class StopOptimization(Exception):
    """Used to signal an optimizer to stop looking for the solution
    and return the control flow to the caller"""

    def __init__(self, reason=StopReason.NOT_GIVEN):
        self.reason = reason
        super().__init__(f"Optimization stopped: {reason.name}")


def check_max_evals(max_evals: int | None, state: HasNumEvals):
    return max_evals is not None and state.num_evaluations >= max_evals


@dataclass
class CMAESEarlyStopping:
    max_evals: int | None = field(default=None)
    tolfun: float | None = field(default=None)

    def check_tolfun(self, state: "CMAESState"):
        if self.tolfun is None or not state.population_evaluations:
            return False

        best, worst = (
            np.min(state.population_evaluations),
            np.max(state.population_evaluations),
        )
        return np.abs(best - worst) < self.tolfun

    def __call__(self, state: "CMAESState", identifier: str = ""):
        maxevals = check_max_evals(self.max_evals, state)
        tolfun = self.check_tolfun(state)

        if maxevals:
            logger.debug(f"[CMA-ES {identifier}] Stopping: reached max evals")
        if tolfun:
            logger.debug(f"[CMA-ES {identifier}] Stopping: tolfun")

        return any((maxevals, tolfun))


@dataclass
class BFGSEarlyStopping:
    max_evals: int | None = field(default=None)

    def __call__(self, state: "BFGSState"):
        if check_max_evals(self.max_evals, state):
            raise StopOptimization(reason=StopReason.MAXEVALS)
        return False
