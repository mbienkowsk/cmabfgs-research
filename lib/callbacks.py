from abc import ABC
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from lib.metrics import Metric
from lib.optimizers.bfgs import BFGSState
from lib.optimizers.cmabfgs import CMABFGSState
from lib.optimizers.cmaes import CMAESState

# TODO: DEDUP


class ExperimentCallback(ABC):
    data: dict[str, list[Any]]

    def export_to_csv(self, path: Path):
        self.as_dataframe().to_csv(path)

    def as_dataframe(self):
        return pd.DataFrame(self.data)


class CMAESMetricsCollector(ExperimentCallback):
    def __init__(self, metrics: Sequence[Metric]):
        self.metrics = metrics
        self.data = {stat.key(): [] for stat in metrics}

    def __call__(self, state: CMAESState):
        for stat in self.metrics:
            self.data[stat.key()].append(stat.collect_cmaes(state))


class BFGSMetricsCollector(ExperimentCallback):
    def __init__(self, metrics: Sequence[Metric]):
        self.metrics = metrics
        self.data = {stat.key(): [] for stat in metrics}

    def __call__(self, state: BFGSState):
        for stat in self.metrics:
            self.data[stat.key()].append(stat.collect_bfgs(state))


class CMABFGSMetricsCollector(ExperimentCallback):
    def __init__(self, metrics: Sequence[Metric]):
        self.metrics = metrics
        self.data = {stat.key(): [] for stat in metrics}

    def __call__(self, state: CMABFGSState):
        for stat in self.metrics:
            self.data[stat.key()].append(stat.collect_cmabfgs(state))
