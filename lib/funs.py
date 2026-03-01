from dataclasses import dataclass
from typing import Callable

import numba
import numpy as np
import sympy as sp

from lib.cec import get_cec2017_for_dim


@dataclass
class OptFun:
    fun: Callable
    grad: Callable | None
    name: str
    optimum: int

    def optimum_for_dim(self, dim: int):
        return np.ones(dim) * self.optimum


def _elliptic(x):
    n = len(x)
    rv = 0
    for i in range(n):
        rv += 10 ** (6 * i / (n - 1)) * x[i] ** 2

    return rv


elliptic = numba.njit(_elliptic)


@numba.njit
def elliptic_grad(x):
    n = len(x)
    rv = np.zeros(n)
    for i in range(n):
        rv[i] = 2 * 10 ** (6 * i / (n - 1)) * x[i]
    return rv


def elliptic_hess_for_dim(dim: int):
    x = sp.symbols(f"x0:{dim}")
    y = _elliptic(x)
    hessian_sym = sp.hessian(y, x)
    # this is a separable quadratic fun, so evaluate it on whatever and return the result
    # as it's constant for all arguments
    return sp.lambdify([x], hessian_sym, modules="numpy")(np.zeros((dim, dim)))


def elliptic_hess_inv_for_dim(dim: int):
    hess = elliptic_hess_for_dim(dim)
    return np.linalg.inv(hess)


Elliptic = OptFun(elliptic, elliptic_grad, "Elliptic", 0)


@numba.njit
def rastrigin(x, A=2.0):
    canonical_bound = 5.12
    external_bound = 100.0
    scale = canonical_bound / external_bound
    y = x * scale

    n = y.size
    result = A * n
    for i in range(n):
        result += y[i] * y[i] - A * np.cos(2.0 * np.pi * y[i])
    return result


Rastrigin = OptFun(rastrigin, None, "Rastrigin", 0)


def get_function_by_name(
    name: str, dim: int = 10, with_optimum: bool = False
) -> Callable | tuple[Callable, int]:
    """Get a function object from a name passed from an env var.
    If a function is prefixed with CEC, it's assumed to be from CEC2017
    with the given number.

    The dim argument only matters for CEC functions.
    """

    if name == "Elliptic":
        return (Elliptic.fun, 0) if with_optimum else Elliptic.fun
    if name == "Rastrigin":
        return (Rastrigin.fun, 0) if with_optimum else Rastrigin.fun
    elif name == "Square":
        return (lambda x: np.sum(x**2), 0) if with_optimum else lambda x: np.sum(x**2)
    elif name.startswith("CEC"):
        try:
            idx = int(name[3:])
            fn = get_cec2017_for_dim(idx, dim)
            return (fn, fn.y_global) if with_optimum else fn  # pyright: ignore[reportReturnType]
        except ValueError:
            raise ValueError(f"Invalid CEC function name: {name}")
        except Exception as e:
            raise RuntimeError(f"Error getting CEC function {name}: {e}")

    else:
        raise ValueError(f"Unknown function name: {name}. ")
