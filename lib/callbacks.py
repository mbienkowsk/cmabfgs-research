from abc import ABC, abstractmethod
from typing import overload

import pandas as pd
from cmaes import CMA
from scipy.optimize import OptimizeResult

from lib.metrics import CMAESMetric
from lib.optimizers.cmaes import CMAESState


class ExperimentCallback(ABC):
    @abstractmethod
    @overload
    def __call__(self, obj: CMA): ...

    @abstractmethod
    @overload
    def __call__(self, obj: OptimizeResult): ...

    @abstractmethod
    def as_dataframe(self) -> pd.DataFrame: ...

    def export_to_csv(self, path: str):
        self.as_dataframe().to_csv(path)


class CMAESMetricsCollector(ExperimentCallback):
    def __init__(self, evalcounter, metrics: list[CMAESMetric]):
        self.evalcounter = evalcounter
        self.metrics = metrics
        self.data = {stat.key(): [] for stat in metrics}

    def __call__(self, state: CMAESState):
        for stat in self.metrics:
            self.data[stat.key()].append(stat.collect(state))

    def as_dataframe(self):
        return pd.DataFrame(self.data)
