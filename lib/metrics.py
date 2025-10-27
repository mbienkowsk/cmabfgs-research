from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from lib.optimizers.bfgs import BFGSState
from lib.optimizers.cmabfgs import CMABFGSState, CMAESState
from lib.optimizers.cmaes import CMAESState
from lib.util import check_bounds


class Metric(ABC):
    @abstractmethod
    def key(self) -> str: ...

    def collect_cmaes(self, state: CMAESState) -> Any:
        raise NotImplementedError

    def collect_bfgs(self, state: BFGSState) -> Any:
        raise NotImplementedError

    def collect_cmabfgs(self, state: CMABFGSState) -> Any:
        if state.mode == "CMAES":
            return self.collect_cmaes(state.cmaes_state)
        elif state.mode == "BFGS":
            return self.collect_bfgs(state.bfgs_state)


class MeanEvaluation(Metric):
    def key(self):
        return "mean"

    def collect_cmaes(self, state: CMAESState):
        return state.counter.without_counting(state.mean)

    def collect_cmabfgs(self, state: CMABFGSState):
        if state.mode == "CMAES":
            return self.collect_cmaes(state.cmaes_state)
        return None


class BestSoFar(Metric):
    def __init__(self, optimum: float):
        # this is needed to store the difference from the global optimum
        # instead of the function value
        self.optimum = optimum

    def key(self):
        return "best"

    def collect_cmaes(self, state: CMAESState):
        return (
            state.best_solutions[-1] - self.optimum if state.best_solutions else pd.NA
        )

    def collect_bfgs(self, state: BFGSState):
        return (
            state.best_solutions[-1] - self.optimum if state.best_solutions else pd.NA
        )


class CovarianceMatrixConditionNumber(Metric):
    def key(self):
        return "cov_cond"

    def collect_cmaes(self, state: CMAESState):
        if state.covariance_matrix is None:
            logger.warning("CMA-ES covariance matrix is None.")
            return pd.NA
        return np.linalg.cond(state.covariance_matrix)


class CovarianceMatrixDifferenceNorm(Metric):
    current_covariance_matrix: np.ndarray | None = None

    def key(self):
        return "cov_diff_sq"

    def collect_cmaes(self, state: CMAESState):
        if state.covariance_matrix is None:
            logger.warning("CMA-ES covariance matrix is None.")
            return pd.NA

        prev = self.current_covariance_matrix
        self.current_covariance_matrix = state.covariance_matrix

        if prev is None:
            return pd.NA

        return np.linalg.norm(self.current_covariance_matrix - prev)


@dataclass
class BoundsCheck(Metric):
    bounds: tuple[int, int]

    def key(self):
        return "in_bounds"

    def collect_cmaes(self, state: CMAESState):
        return check_bounds(state.mean, self.bounds)

    def collect_bfgs(self, state: BFGSState):
        if not state.current_result:
            return pd.NA
        return check_bounds(state.current_result.x, self.bounds)
