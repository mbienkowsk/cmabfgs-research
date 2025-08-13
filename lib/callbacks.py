from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd

from lib.metrics import Metric
from lib.util import EvalCounter


class HasCounter(Protocol):
    counter: EvalCounter


@dataclass
class MetricsCollector:
    metrics: Sequence[Metric]
    collect_method: str
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __call__(self, state: HasCounter):
        evals = state.counter.num_evaluations
        entry = {"num_evaluations": [evals]}
        for metric in self.metrics:
            collect_fn = getattr(metric, f"collect_{self.collect_method}")
            entry[metric.key()] = [collect_fn(state)]

        entry_df = pd.DataFrame(entry).dropna()
        if self.data.empty:
            self.data = entry_df
        else:
            self.data = pd.concat([self.data, entry_df])

    def as_dataframe(self):
        self.data.set_index("num_evaluations", inplace=True)
        return self.data

    def export_to_csv(self, path: Path):
        self.as_dataframe().to_csv(path)
