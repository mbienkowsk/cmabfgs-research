from abc import ABC, abstractmethod
from typing import Any

from lib.optimizers.bfgs import BFGSState
from lib.optimizers.cmaes import CMAESState


class Metric(ABC):
    @abstractmethod
    def key(self) -> str: ...

    def collect_cmaes(self, state: CMAESState) -> Any:
        raise NotImplementedError

    def collect_bfgs(self, state: BFGSState) -> Any:
        raise NotImplementedError


class MeanEvaluation(Metric):
    def key(self):
        return "mean"

    def collect_cmaes(self, state: CMAESState):
        return state.counter.without_counting(state.mean)


class BestSoFar(Metric):
    def key(self):
        return "best"

    def collect_cmaes(self, state: CMAESState):
        return state.best_solutions[-1] if state.best_solutions else None

    def collect_bfgs(self, state: BFGSState):
        return state.best_solutions[-1] if state.best_solutions else None
