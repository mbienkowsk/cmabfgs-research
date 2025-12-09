from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from lib.optimizers.bfgs import BFGSState
from lib.optimizers.cmaes import CMAESState
from lib.optimizers.golden_search import GoldenSearchState
from lib.optimizers.hybrids.cmabfgs import CMABFGSState
from lib.optimizers.lbfgs import L_BFGS_BState
from lib.util import EvalCounter


class Metric(ABC):
    @abstractmethod
    def key(self) -> str: ...

    def collect_cmaes(self, state: CMAESState) -> Any:
        raise NotImplementedError

    def collect_bfgs(self, state: BFGSState) -> Any:
        raise NotImplementedError

    def collect_from_counter(self, counter: EvalCounter) -> Any:
        raise NotImplementedError

    def collect_cmabfgs(self, state: CMABFGSState) -> Any:
        if state.mode == "CMAES":
            return self.collect_cmaes(state.cmaes_state)
        elif state.mode == "BFGS":
            return self.collect_bfgs(state.bfgs_state)

    def collect(self, state: CMAESState | BFGSState | GoldenSearchState) -> Any:
        if isinstance(state, CMAESState):
            return self.collect_cmaes(state)
        elif isinstance(state, (L_BFGS_BState, BFGSState)):
            return self.collect_bfgs(state)
        elif isinstance(state, GoldenSearchState):
            return self.collect_from_counter(state.counter)


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
    def __init__(self, optimum: float | None = None):
        # this is needed to store the difference from the global optimum
        # instead of the function value
        self.optimum = optimum if optimum is not None else 0

    def key(self):
        return "best"

    def collect_cmaes(self, state: CMAESState):
        return state.counter.best_so_far - self.optimum

    def collect_bfgs(self, state: BFGSState):
        return state.counter.best_so_far - self.optimum

    def collect_from_counter(self, counter: EvalCounter):
        return counter.best_so_far - self.optimum


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


class SigmaMeasurement(Metric):
    def key(self):
        return "sigma"

    def collect_cmaes(self, state: CMAESState):
        return state.sigma


class CovarianceMatrixEigenvalueList(Metric):
    def key(self):
        return "cov_mat_eigv"

    def collect_cmaes(self, state: CMAESState):
        eigenvalues, _ = np.linalg.eigh(state.covariance_matrix)
        return eigenvalues
