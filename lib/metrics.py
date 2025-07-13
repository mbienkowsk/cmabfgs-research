from abc import ABC, abstractmethod
from typing import Any

from cmaes import CMA

from lib.util import EvalCounter


class Metric(ABC):
    @abstractmethod
    def key(self) -> str: ...
    @abstractmethod
    def collect(self, cmaes: CMA, evalcounter: EvalCounter) -> Any: ...


class CMAESMetric(Metric): ...


class MeanEvaluation(CMAESMetric):
    def key(self):
        return "mean"

    def collect(self, cmaes, evalcounter):
        return evalcounter.fun(cmaes.mean)


class BestSoFar(CMAESMetric):
    def key(self):
        return "best"

    def collect(self, cmaes, evalcounter):
        return evalcounter.best_solutions[-1] if evalcounter.best_solutions else None
