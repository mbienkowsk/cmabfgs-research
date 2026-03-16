from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from lib.optimizers.bfgs import BFGSState
from lib.optimizers.cmaes import CMAESState
from lib.optimizers.golden_search import GoldenSearchState
from lib.optimizers.lbfgs import L_BFGS_BState
from lib.util import EvalCounter


class Metric(ABC):
    @abstractmethod
    def key(self) -> str: ...

    def collect_cmaes(self, state: CMAESState) -> Any:
        raise NotImplementedError

    def collect_bfgs(self, state: BFGSState | L_BFGS_BState) -> Any:
        raise NotImplementedError

    def collect_from_counter(self, counter: EvalCounter) -> Any:
        raise NotImplementedError

    def collect(self, state: CMAESState | BFGSState | GoldenSearchState) -> Any:
        if isinstance(state, CMAESState):
            return self.collect_cmaes(state)
        elif isinstance(state, (L_BFGS_BState, BFGSState)):
            return self.collect_bfgs(state)
        elif isinstance(state, GoldenSearchState):
            return self.collect_from_counter(state.counter)


class Mean(Metric):
    def key(self):
        return "mean"

    def collect_cmaes(self, state: CMAESState):
        return deepcopy(state.mean)


class MeanEvaluation(Metric):
    def key(self):
        return "fmean"

    def collect_cmaes(self, state: CMAESState):
        return state.counter.without_counting(state.mean)


class BestSoFar(Metric):
    def __init__(self, optimum: float | None = None):
        # this is needed to store the difference from the global optimum
        # instead of the function value
        self.optimum = optimum if optimum is not None else 0

    def key(self):
        return "best"

    def collect_cmaes(self, state: CMAESState):
        return state.counter.best_so_far[1] - self.optimum

    def collect_bfgs(self, state: BFGSState | L_BFGS_BState):
        return state.counter.best_so_far[1] - self.optimum

    def collect_from_counter(self, counter: EvalCounter):
        return counter.best_so_far[1] - self.optimum


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


@dataclass
class CovarianceMatrix(Metric):
    # whether to flatten and convert to list so that it can be read back
    # from parquet
    serialize: bool = False

    def key(self):
        return "cov_mat"

    def collect_cmaes(self, state: CMAESState):
        C = deepcopy(state.covariance_matrix)
        return list(np.ravel(C)) if self.serialize else C


@dataclass
class CovarianceMatrixNorm(Metric):
    def key(self):
        return "cov_mat_norm"

    def collect_cmaes(self, state: CMAESState):
        return np.linalg.norm(state.covariance_matrix)


class BestXSoFar(Metric):
    def key(self):
        return "xbest"

    def collect_cmaes(self, state: CMAESState):
        return state.counter.best_so_far[0]


@dataclass
class CMAESIteration(Metric):
    popsize: int

    def key(self):
        return "iteration"

    def collect_cmaes(self, state: CMAESState):
        return state.num_evaluations // self.popsize


@dataclass
class CMAESPopulation(Metric):
    def key(self):
        return "population"

    def collect_cmaes(self, state: CMAESState):
        return state.population
