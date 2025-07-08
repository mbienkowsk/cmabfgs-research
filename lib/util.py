from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, overload

import numpy as np
from scipy.optimize import OptimizeResult

from lib.optimizers import Optimizer


def gradient_central(func: Callable, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    n = len(x)
    grad = np.zeros_like(x, dtype=float)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += h
        x_minus[i] -= h

        grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)

    return grad


class ExperimentCallback(ABC):
    @abstractmethod
    @overload
    def __call__(self, obj: Optimizer): ...

    @abstractmethod
    @overload
    def __call__(self, intermediate_result: OptimizeResult): ...


@dataclass
class Experiment(ABC):
    callbacks: list[ExperimentCallback]


@dataclass
class EvalCounter:
    """A wrapper to count the number of evaluations & keep track of the
    best solution"""

    fun: Callable
    num_evaluations: int = field(default=0)
    best_solution: tuple[np.ndarray, float] | None = field(default=None)

    def __call__(self, x):
        self.num_evaluations += 1
        y = self.fun(x)

        if self.best_solution is None or y < self.best_solution[1]:
            self.best_solution = (x.copy(), y)

        return y


class StopOptimization(Exception):
    """Used to signal an optimizer to stop looking for the solution
    and return the control flow to the caller"""


def one_dim(fun: Callable, x, d):
    """Gimmick to make a multdimensional function 1dim
    with a set direction d"""

    def wrapper(alpha):
        return fun(x + alpha * d)

    return wrapper
