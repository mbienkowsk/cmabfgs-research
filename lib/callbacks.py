from abc import ABC
from typing import Any

import pandas as pd

from lib.metrics import Metric
from lib.optimizers.bfgs import BFGSState
from lib.optimizers.cmaes import CMAESState


class ExperimentCallback(ABC):
    data: dict[str, list[Any]]

    def export_to_csv(self, path: str):
        self.as_dataframe().to_csv(path)

    def as_dataframe(self):
        return pd.DataFrame(self.data)


class CMAESMetricsCollector(ExperimentCallback):
    def __init__(self, metrics: list[Metric]):
        self.metrics = metrics
        self.data = {stat.key(): [] for stat in metrics}

    def __call__(self, state: CMAESState):
        for stat in self.metrics:
            self.data[stat.key()].append(stat.collect_cmaes(state))


class BFGSMetricsCollector(ExperimentCallback):
    def __init__(self, metrics: list[Metric]):
        self.metrics = metrics
        self.data = {stat.key(): [] for stat in metrics}

    def __call__(self, state: BFGSState):
        for stat in self.metrics:
            self.data[stat.key()].append(stat.collect_bfgs(state))
