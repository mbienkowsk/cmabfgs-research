import fcntl
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


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


@dataclass
class EvalCounter:
    """A wrapper to count the number of evaluations & keep track of the
    best solution"""

    @property
    def without_counting(self):
        return self.fun

    fun: Callable
    num_evaluations: int = field(default=0)
    best_solutions: list[float] = field(default_factory=list)

    def __call__(self, x):
        self.num_evaluations += 1
        y = self.fun(x)

        if y < 0:
            with open("negative_eval.log", "a") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.write(f"Negative evaluation: {y} at {x}\n")
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        if not self.best_solutions or y < self.best_solutions[-1]:
            self.best_solutions.append(y)
        else:
            self.best_solutions.append(self.best_solutions[-1])

        return y


def one_dimensional(fun: Callable, x, d):
    """Gimmick to make a multdimensional function 1dim
    with a set direction d"""

    def wrapper(alpha):
        return fun(x + alpha * d)

    return wrapper


def extract_dim_from_path(path: Path):
    """Extracts the dimension from a path containing 'DIM_<number>'."""
    match = re.search(r"DIM_(\d+)", str(path).upper())
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract dimension from path: {path}")


def extract_objective_from_path(path: Path):
    """Extracts the objective function name from a path containing 'FUN_<name>_'."""
    match = re.search(r"FUN_([^_]+)", str(path).upper())
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract objective from path: {path}")
