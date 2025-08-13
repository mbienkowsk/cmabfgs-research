from dataclasses import dataclass
from typing import Callable

import numba
import numpy as np

from lib.cec import get_cec2017_for_dim


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

    elif name.startswith("CEC"):
        try:
            idx = int(name[3:])
            fn = get_cec2017_for_dim(idx, dim)
            return (
                (fn.evaluate, fn.f_global) if with_optimum else fn.evaluate
            )  # pyright: ignore[reportReturnType]
        except ValueError:
            raise ValueError(f"Invalid CEC function name: {name}")
        except Exception as e:
            raise RuntimeError(f"Error getting CEC function {name}: {e}")

    else:
        raise ValueError(f"Unknown function name: {name}. ")
