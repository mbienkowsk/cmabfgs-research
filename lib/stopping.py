from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np
from cmaes import CMA

from lib.util import EvalCounter


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
class EarlyStopping(ABC):
    @abstractmethod
    def check(self, context: Any):
        """Check whether to stop the optimizer and throw StopOptimization if so"""


@dataclass
class MaxEvalsStopping(EarlyStopping):
    """Base class for early stopping based on maximum number of evaluations."""


@dataclass
class TolFunStopping(EarlyStopping):
    """Base class for early stopping based on the toleration for the difference between the best and
    worst individual in a population."""


@dataclass
class TolHistStopping(EarlyStopping):
    # TODO: implement
    """Base class for early stopping based on history tolerance."""


@dataclass
class CMAESContext:
    cmaes: CMA
    eval_counter: EvalCounter
    fitness_values: np.ndarray


@dataclass
class CMAESMaxEvalsStopping(MaxEvalsStopping):
    max_evals: int

    def check(self, context: CMAESContext):
        if context.eval_counter.num_evaluations >= self.max_evals:
            raise StopOptimization(StopReason.MAXEVALS)


@dataclass
class TolFunCMAESStopping(TolFunStopping):
    tolerance: float

    def check(self, context: CMAESContext):
        best, worst = np.min(context.fitness_values), np.max(context.fitness_values)
        return abs(best - worst) < self.tolerance
