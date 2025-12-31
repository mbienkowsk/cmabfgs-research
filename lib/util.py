import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Hashable, cast

import numpy as np
import pandas as pd
from loguru import logger
from sympy import prime

from lib.bound_handling import check_bounds


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
    best_solutions: list[tuple[np.ndarray | None, float]] = field(default_factory=list)
    bounds: tuple[float, float] | None = None
    identifier: str = ""

    def __call__(self, x):
        self.num_evaluations += 1

        y = self.fun(x)
        xbest, ybest = self.best_so_far

        if self.bounds and not check_bounds(x, self.bounds, False):
            msg = (
                "Out of bounds evaluation detected. Refusing to update best_solutions."
            )
            if self.identifier:
                msg = f"{self.identifier}: {msg}"
            logger.warning(msg)
            self.best_solutions.append((xbest, ybest))
            return y

        if not self.best_solutions or y < ybest:
            self.best_solutions.append((x, y))
        else:
            self.best_solutions.append((xbest, ybest))

        return y

    @property
    def best_so_far(self):
        return self.best_solutions[-1] if self.best_solutions else (None, np.inf)

    def copy_with_identifier(self, identifier: str):
        return EvalCounter(
            self.fun,
            self.num_evaluations,
            # NOTE: can't deepcopy cecxx as they're not serializable yet,
            # things have to be manually deepcopied here to avoid sharing lists!
            self.best_solutions[:],
            self.bounds,
            identifier,
        )


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


def assert_non_increasing(data_storage: pd.Series | pd.DataFrame, msg: str = ""):
    assert not (data_storage.diff() > 0).any().any(), (
        msg or "Series is not non-increasing"
    )


def assert_all_non_increasing(
    data_containers: Iterable[pd.Series | pd.DataFrame], msg: str = ""
):
    for c in data_containers:
        assert_non_increasing(c, msg)


def get_x0_and_seed_for_run_id(run_id: int, dimensions: int, bounds: int):
    """Boilerplate for setting up a random generator based on the run_id-th prime and returning a starting point for the given bounds and the prime (seed)"""
    seed: int = prime(run_id)  # pyright: ignore[reportAssignmentType]
    rng = np.random.default_rng(seed)
    x = cast(
        np.ndarray,  # pyright: ignore[reportArgumentType]
        (rng.random(dimensions) - 0.5) * 2 * bounds,  # pyright: ignore[reportArgumentType]
    )
    return x, seed


def compress_and_save(df: pd.DataFrame, path: Path):
    df.to_parquet(path, index=True, compression="brotli")


def summarize_data(df: pd.DataFrame):
    print("Data Summary")
    print("=" * 40)

    def is_hashable_series(s):
        return s.dropna().apply(lambda x: isinstance(x, Hashable)).all()

    hashable_cols = [c for c in df.columns if is_hashable_series(df[c])]  # pyright: ignore[reportGeneralTypeIssues]

    unique_counts = df[hashable_cols].nunique(dropna=True)

    summary = pd.DataFrame(
        {
            "dtype": df.dtypes,
            "non_null": df.count(),
            "nulls": df.isna().sum(),
        }
    )

    summary["unique"] = np.nan
    summary.loc[hashable_cols, "unique"] = unique_counts

    num_cols = df.select_dtypes(include="number")
    if not num_cols.empty:
        summary.loc[num_cols.columns, "min"] = num_cols.min()
        summary.loc[num_cols.columns, "max"] = num_cols.max()
        summary.loc[num_cols.columns, "mean"] = num_cols.mean()

    print(summary)

    print("\nHead")
    print("-" * 40)
    print(df.head())

    print("\nTail")
    print("-" * 40)
    print(df.tail())
