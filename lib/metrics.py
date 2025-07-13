from abc import ABC, abstractmethod
from typing import Any

from lib.optimizers.cmaes import CMAESState


class Metric(ABC):
    @abstractmethod
    def key(self) -> str: ...
    @abstractmethod
    def collect(self, state: CMAESState) -> Any: ...


class CMAESMetric(Metric): ...


class MeanEvaluation(CMAESMetric):
    def key(self):
        return "mean"

    def collect(self, state: CMAESState):
        return state.counter.without_counting(state.mean)


class BestSoFar(CMAESMetric):
    def key(self):
        return "best"

    def collect(self, state: CMAESState):
        return state.best_solutions[-1] if state.best_solutions else None
