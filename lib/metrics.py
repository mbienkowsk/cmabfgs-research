from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from loguru import logger

from lib.optimizers.bfgs import BFGSState
from lib.optimizers.cmabfgs import CMABFGSState, CMAESState
from lib.optimizers.cmaes import CMAESState


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
    def key(self):
        return "best"

    def collect_cmaes(self, state: CMAESState):
        return state.best_solutions[-1] if state.best_solutions else None

    def collect_bfgs(self, state: BFGSState):
        return state.best_solutions[-1] if state.best_solutions else None


class CovarianceMatrixConditionNumber(Metric):
    def key(self):
        return "cov_cond"

    def collect_cmaes(self, state: CMAESState):
        if state.covariance_matrix is None:
            logger.warning("CMA-ES covariance matrix is None.")
            return
        return np.linalg.cond(state.covariance_matrix)


class CovarianceMatrixDifferenceNorm(Metric):
    current_covariance_matrix: np.ndarray | None = None

    def key(self):
        return "cov_diff_sq"

    def collect_cmaes(self, state: CMAESState):
        if state.covariance_matrix is None:
            logger.warning("CMA-ES covariance matrix is None.")
            return 0

        prev = self.current_covariance_matrix
        self.current_covariance_matrix = state.covariance_matrix

        if prev is None:
            return 0

        return np.linalg.norm(self.current_covariance_matrix - prev)
