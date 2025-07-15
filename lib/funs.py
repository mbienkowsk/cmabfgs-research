from dataclasses import dataclass
from typing import Callable

import numba
import numpy as np


@dataclass
class OptFun:
    fun: Callable
    grad: Callable
    name: str
    optimum: int

    def optimum_for_dim(self, dim: int):
        return np.ones(dim) * self.optimum


@numba.njit
def elliptic(x):
    n = len(x)
    rv = 0
    for i in range(n):
        rv += 10 ** (6 * i / (n - 1)) * x[i] ** 2

    return rv


@numba.njit
def elliptic_grad(x):
    n = len(x)
    rv = np.zeros(n)
    for i in range(n):
        rv[i] = 2 * 10 ** (6 * i / (n - 1)) * x[i]
    return rv


Elliptic = OptFun(elliptic, elliptic_grad, "Elliptic", 0)
